// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <vector>
#include <magic_enum/magic_enum.hpp>

#include "common/alignment.h"
#include "core/debug_state.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;

static constexpr auto ARENA_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

std::optional<u32> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                  vk::MemoryPropertyFlags wanted, u32 memory_type_bits) {
    for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
        if (((memory_type_bits >> i) & 1) == 0) {
            continue;
        }
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_, PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      memory_semaphore{instance} {
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = ARENA_PAGE_SIZE,
        .usage = ARENA_USAGE,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const vk::DeviceBufferMemoryRequirements req_info = {
        .pCreateInfo = &probe_ci,
    };
    const auto device = instance.GetDevice();
    const auto reqs = device.getBufferMemoryRequirements(req_info).memoryRequirements;
    block_size = Common::AlignUp(std::max<u64>(reqs.alignment, MIN_BLOCK_SIZE), reqs.alignment);
    ASSERT_MSG(std::popcount(block_size) == 1, "Sparse block size {} is not a power of 2",
               block_size);
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = ARENA_PAGE_SIZE / block_size;
    blocks_per_arena_page_shift = ARENA_PAGE_BITS - block_shift;
    arena_memory_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();

    const u64 bda_pagetable_size =
        (blocks_per_arena_page * NUM_ARENA_PAGES) * sizeof(vk::DeviceAddress);
    fault_manager = std::make_unique<FaultManager>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * NUM_ARENA_PAGES);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);
}

BufferCache::~BufferCache() = default;

void BufferCache::TickFrame() {
    if (std::exchange(fault_process_pending, false)) {
        fault_manager->ProcessFaultBuffer();
    }
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    memory_tracker->InvalidateRegion(device_addr, size, [this, device_addr, size, assume_locks] {
        ReadMemory(device_addr, size, true, assume_locks);
    });
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool assume_locks) {
    // The GPU thread only records the copy and submits it. The faulting thread waits, so command
    // processing continues while the GPU finishes the readback.
    auto finish = [this, device_addr, size, is_write](ReadbackTicket ticket) {
        if (ticket.valid) {
            FinishReadback(std::move(ticket));
        }
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    };
    if (assume_locks) {
        auto ticket = RecordReadback(device_addr, size);
        if (ticket.valid) {
            ticket.tick = scheduler.CurrentTick();
            scheduler.Flush();
            scheduler.Wait(ticket.tick);
        }
        finish(std::move(ticket));
        return;
    }

    ReadbackTicket ticket;
    liverpool->SendCommand<true>([&] {
        ticket = RecordReadback(device_addr, size);
        if (ticket.valid) {
            ticket.tick = scheduler.CurrentTick();
            scheduler.Flush();
        } else if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    });
    if (!ticket.valid) {
        return;
    }
    scheduler.WaitSubmitted(ticket.tick);
    liverpool->SendCommand<true>([&] { finish(std::move(ticket)); });
}

BufferCache::ReadbackTicket BufferCache::RecordReadback(VAddr device_addr, u64 size) {
    // One GPU wait for the fault window and every other dirty range that fits the budget.
    // The copied bytes match a precise readback; later GPU writes re-arm read tracking.
    constexpr u64 WindowSize = 512_KB;
    constexpr u64 CoalesceBudget = 32_MB;
    constexpr u64 StagingAlign = 64;

    const u32 first_block = device_addr >> block_shift;
    const u32 last_block = (device_addr + size - 1) >> block_shift;
    const auto* fault_arena = GetArena(first_block, last_block);
    const VAddr arena_end = fault_arena->cpu_addr + fault_arena->size_bytes;
    const VAddr window_start =
        std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), fault_arena->cpu_addr);
    const VAddr window_end = std::min<VAddr>(
        std::max<VAddr>(window_start + WindowSize, device_addr + size), arena_end);

    struct Window {
        VAddr addr;
        u64 size;
    };
    std::vector<Window> windows;
    windows.push_back({window_start, window_end - window_start});
    u64 budget = window_end - window_start;

    std::vector<std::pair<VAddr, VAddr>> modified;
    gpu_modified_ranges.ForEach([&](VAddr begin, VAddr end) { modified.emplace_back(begin, end); });
    const auto add_window = [&](VAddr begin, VAddr end) {
        if (begin >= end || budget >= CoalesceBudget) {
            return;
        }
        u64 span = end - begin;
        if (budget + span > CoalesceBudget) {
            span = CoalesceBudget - budget;
        }
        windows.push_back({begin, span});
        budget += span;
    };
    for (const auto [begin, end] : modified) {
        if (budget >= CoalesceBudget) {
            break;
        }
        if (begin < window_end && end > window_start) {
            if (begin < window_start) {
                add_window(begin, window_start);
            }
            if (end > window_end) {
                add_window(window_end, end);
            }
        } else {
            add_window(begin, end);
        }
    }

    struct Piece {
        const Buffer* arena;
        VAddr src;
        u64 size;
        u64 pack_off;
    };
    RangeSet covered;
    std::vector<Piece> pieces;
    u64 total_size_bytes = 0;
    const auto add_piece = [&](const Buffer* arena, VAddr src, u64 piece_size) {
        if (piece_size == 0) {
            return;
        }
        pieces.push_back({arena, src, piece_size, total_size_bytes});
        total_size_bytes += (piece_size + StagingAlign - 1) & ~(StagingAlign - 1);
    };

    for (const Window& window : windows) {
        VAddr addr = window.addr;
        u64 remaining = window.size;
        while (remaining > 0) {
            const u64 page_off = addr & (ARENA_PAGE_SIZE - 1);
            const u64 slice = std::min(remaining, ARENA_PAGE_SIZE - page_off);
            const u64 slice_first = addr >> block_shift;
            const u64 slice_last = (addr + slice - 1) >> block_shift;
            const Buffer* arena = GetArena(slice_first, slice_last);
            memory_tracker->ForEachDownloadRange<false>(addr, slice, [&](u64 address, u64 range_size) {
                gpu_modified_ranges.ForEachInRange(
                    address, range_size, [&](VAddr start, VAddr end) {
                        boost::container::small_vector<std::pair<VAddr, u64>, 4> gaps;
                        covered.ForEachNotInRange(start, end - start, [&](VAddr gap, u64 gap_size) {
                            gaps.emplace_back(gap, gap_size);
                        });
                        for (const auto [gap, gap_size] : gaps) {
                            add_piece(arena, gap, gap_size);
                            covered.Add(gap, gap_size);
                        }
                    });
            });
            addr += slice;
            remaining -= slice;
        }
    }

    if (pieces.empty()) {
        return {};
    }

    ReadbackTicket ticket;
    ticket.valid = true;
    ticket.window = window_start;
    ticket.window_size = window_end - window_start;
    ticket.writes.reserve(pieces.size());
    ticket.download =
        staging_pool.Request(total_size_bytes, MemoryType::HostCached, StagingAlign, true);

    scheduler.SuspendUploadFlush();
    for (size_t index = 0; index < pieces.size();) {
        const Buffer* arena = pieces[index].arena;
        boost::container::small_vector<vk::BufferCopy, 8> copies;
        for (; index < pieces.size() && pieces[index].arena == arena; ++index) {
            const Piece& piece = pieces[index];
            copies.push_back(vk::BufferCopy{
                .srcOffset = piece.src - arena->cpu_addr,
                .dstOffset = ticket.download.offset + piece.pack_off,
                .size = piece.size,
            });
            ticket.writes.push_back({piece.src, piece.pack_off, piece.size});
        }
        runtime.CopyBuffer(arena, ticket.download.buffer, copies);
    }
    scheduler.ResumeUploadFlush();
    return ticket;
}

void BufferCache::FinishReadback(ReadbackTicket ticket) {
    if (!ticket.valid) {
        return;
    }
    ticket.download.Invalidate();
    for (const auto& write : ticket.writes) {
        memory->TryWriteBacking(std::bit_cast<u8*>(write.dst),
                                ticket.download.mapped + write.pack_off, write.size);
        memory_tracker->UnmarkRegionAsGpuModified(write.dst, write.size, false);
        gpu_modified_ranges.Subtract(write.dst, write.size);
    }
    memory_tracker->UnmarkRegionAsGpuModified(ticket.window, ticket.window_size, false);
    staging_pool.FreeDeferred(ticket.download);
}

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        bool is_written, bool is_texel_buffer) {
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && size <= STREAM_THRESHOLD && !IsRegionGpuModified(device_addr, size)) {
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        return {&stream_buffer, offset};
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    QueueUpload(arena, device_addr, size, is_written);
    if (is_texel_buffer && !is_written) {
        pending_texels.push_back({arena, device_addr, size});
        FlushPendingUploads();
    }
    if (is_written) {
        gpu_modified_ranges.Add(device_addr, size);
    }
    return {arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    if (IsRegionGpuModified(device_addr, size)) {
        return ObtainBuffer(device_addr, size, false);
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    staging.Flush();
    return {staging.buffer, staging.offset};
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

void BufferCache::SynchronizeDmaBuffers() {
    FlushPendingUploads();
    fault_process_pending = true;
    for (const auto& range : resident_ranges) {
        const u64 page = range.start >> (ARENA_PAGE_BITS - block_shift);
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(address_space[page], device_addr, size, false, false);
    }
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    ASSERT_MSG(last_page - first_page <= 1,
               "Buffer request cannot span more than two VA arena pages");

    const auto* first_arena = address_space[first_page];
    const auto* last_arena = address_space[last_page];
    if (first_arena == last_arena) {
        if (!first_arena) {
            const u64 base_block = Common::AlignDownPow2<u64>(first_block, blocks_per_arena_page);
            const u64 num_pages = last_page - first_page + 1;
            const auto* new_arena =
                &arenas.emplace_back(instance, base_block << block_shift,
                                     num_pages << ARENA_PAGE_BITS, MemoryType::Sparse);
            address_space[first_page] = new_arena;
            address_space[last_page] = new_arena;
        }
        return address_space[first_page];
    }

    LOG_WARNING(Render, "Migrating arena");

    const u64 first_addr = first_arena ? first_arena->cpu_addr : (first_page << ARENA_PAGE_BITS);
    const u64 first_size = first_arena ? first_arena->size_bytes : ARENA_PAGE_SIZE;
    const u64 last_size = last_arena ? last_arena->size_bytes : ARENA_PAGE_SIZE;

    const u64 base_block = first_addr >> block_shift;
    const u64 total_size = first_size + last_size;
    const u64 end_block = (first_addr + total_size) >> block_shift;
    auto* new_arena = &arenas.emplace_back(instance, first_addr, total_size, MemoryType::Sparse);
    auto* bind = BindsForArena(new_arena);
    resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(base_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        bind->binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = (start - base_block) << block_shift,
            .size = (end - start) << block_shift,
            .memory = backing.memory,
            .memoryOffset = backing.offset + ((start - backing.start) << block_shift),
        });
    });

    u64 base_page = first_addr >> ARENA_PAGE_BITS;
    for (u32 page = 0; page < (first_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    base_page = last_page;
    for (u32 page = 0; page < (last_size >> ARENA_PAGE_BITS); ++page) {
        address_space[base_page + page] = new_arena;
    }
    return new_arena;
}

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    u32 resident_blocks{};
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64 start, u64 end) {
        resident_blocks += end - start;
        bind_ranges.Add({start, end});
    });

    if (bind_ranges.Empty()) {
        return;
    }

    const vk::MemoryAllocateInfo alloc_info = {
        .allocationSize = resident_blocks << block_shift,
        .memoryTypeIndex = arena_memory_type_index,
    };
    const auto device_memory = Vulkan::Check(instance.GetDevice().allocateMemory(alloc_info));

    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        staging_pool.Request(resident_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);

    u64 memory_offset{};
    ArenaBinds* binds = BindsForArena(arena);
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& range : bind_ranges) {
        Backing backing;
        backing.start = range.start;
        backing.end = range.end;
        backing.memory = device_memory;
        backing.offset = memory_offset;
        resident_ranges.Add(backing);

        LOG_INFO(Render, "Making range start={}, end={} resident", backing.start, backing.end);

        const auto& bind = binds->binds.emplace_back(vk::SparseMemoryBind{
            .resourceOffset = (range.start << block_shift) - arena->cpu_addr,
            .size = (range.end - range.start) << block_shift,
            .memory = device_memory,
            .memoryOffset = memory_offset,
        });
        memory_offset += bind.size;

        for (u32 block = 0; block < bind.size; block += block_size) {
            *(bda_addrs++) = arena->BufferDeviceAddress() + bind.resourceOffset + block;
        }
        const u64 copy_size = (backing.end - backing.start) * sizeof(vk::DeviceAddress);
        copies.emplace_back(offset, backing.start * sizeof(vk::DeviceAddress), copy_size);
        offset += copy_size;
    }

    staging.Flush();
    runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
}

void BufferCache::QueueUpload(const Buffer* arena, VAddr device_addr, u32 size, bool is_written) {
    memory_tracker->ForEachUploadRange(device_addr, size, is_written, [&](u64 addr, u64 range_size) {
        const u64 pack_off = pending_upload_bytes.size();
        pending_upload_bytes.resize(pack_off + range_size);
        memory->CopySparseMemory(addr, pending_upload_bytes.data() + pack_off, range_size);
        pending_uploads.push_back(PendingUpload{
            .arena = arena,
            .pack_off = pack_off,
            .dst_offset = addr - arena->cpu_addr,
            .size = range_size,
        });
    });
}

void BufferCache::FlushPendingUploads() {
    if (flushing_uploads || (pending_uploads.empty() && pending_texels.empty())) {
        return;
    }
    flushing_uploads = true;
    const auto uploads = std::move(pending_uploads);
    const auto bytes = std::move(pending_upload_bytes);
    const auto texels = std::move(pending_texels);
    pending_uploads.clear();
    pending_upload_bytes.clear();
    pending_texels.clear();

    if (!uploads.empty()) {
        const auto staging =
            staging_pool.Request(bytes.size(), MemoryType::HostUncached, 0, true);
        std::memcpy(staging.mapped, bytes.data(), bytes.size());
        staging.Flush();

        std::vector<u8> emitted(uploads.size(), 0);
        for (size_t index = 0; index < uploads.size(); ++index) {
            if (emitted[index]) {
                continue;
            }
            const Buffer* arena = uploads[index].arena;
            boost::container::small_vector<vk::BufferCopy, 8> copies;
            for (size_t cursor = index; cursor < uploads.size(); ++cursor) {
                if (emitted[cursor] || uploads[cursor].arena != arena) {
                    continue;
                }
                emitted[cursor] = 1;
                const PendingUpload& upload = uploads[cursor];
                copies.push_back(vk::BufferCopy{
                    .srcOffset = staging.offset + upload.pack_off,
                    .dstOffset = upload.dst_offset,
                    .size = upload.size,
                });
            }
            runtime.CopyBuffer(staging.buffer, arena, copies);
        }
        staging_pool.FreeDeferred(staging);

        const vk::MemoryBarrier2 barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        };
        const vk::DependencyInfo dependency = {
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &barrier,
        };
        scheduler.CommandBuffer().pipelineBarrier2(dependency);
    }

    for (const PendingTexel& texel : texels) {
        SynchronizeMemoryFromImage(texel.arena, texel.addr, texel.size);
    }
    flushing_uploads = false;
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    memory_tracker->ForEachUploadRange(device_addr, size, is_written, [&](u64 addr, u64 size) {
        copies.emplace_back(total_size_bytes, addr, size);
        total_size_bytes += size;
    });
    if (!copies.empty()) {
        const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
        for (auto& copy : copies) {
            memory->CopySparseMemory(copy.dstOffset, staging.mapped + copy.srcOffset, copy.size);
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
        }
        staging.Flush();
        runtime.CopyBuffer(staging.buffer, arena, copies);
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeMemoryFromImage(arena, device_addr, size);
    }
    return false;
}

bool BufferCache::SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size) {
    if (auto type = texture_cache.IsMeta(device_addr)) {
        if (*type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}", magic_enum::enum_name(*type));
        }
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u64 arena_offset = arena->Offset(device_addr);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (arena_offset + mip_info.offset + mip_info.size > arena->size_bytes) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
    }
    if (buffer_copies.empty()) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, arena, arena_offset);
    return true;
}

std::optional<PendingSparseBatch> BufferCache::TakePendingArenaBinds(Vulkan::SubmitInfo& info) {
    if (pending_binds.empty()) {
        return std::nullopt;
    }

    PendingSparseBatch batch;
    batch.buffers.reserve(pending_binds.size());
    for (const auto& binds : pending_binds) {
        PendingSparseBuffer buffer;
        buffer.buffer = binds.arena->Handle();
        buffer.binds.assign(binds.binds.begin(), binds.binds.end());
        batch.buffers.push_back(std::move(buffer));
    }

    batch.signal_tick = memory_semaphore.NextTick();
    batch.signal = memory_semaphore.Handle();
    info.AddWait(batch.signal, batch.signal_tick);
    pending_binds.clear();
    return batch;
}

} // namespace VideoCore
