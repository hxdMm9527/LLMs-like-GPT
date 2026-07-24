import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    # 将文件分块，使每个块可以独立统计。
    # 如果边界最终重叠，可能返回的分块数少于期望值。
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    # 特殊 token 必须以字节串表示
    # Must represent special token as a bytestring
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # 获取文件的总字节大小
    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # 均匀间隔的分块边界位置的初始猜测
    # 分块从上一个索引开始，不包括最后一个索引
    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # 每次向前读取 4k 字节
    # Read ahead by 4k bytes at a time
    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # 从边界猜测处开始
        # Start at boundary guess
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)  # 读取一个小块
            # Read a mini chunk
            mini_chunk = file.read(mini_chunk_size)

            # 如果到了文件末尾，此边界应该在文件末尾
            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在小块中找到特殊 token
            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # 确保所有边界是唯一的，但可能少于期望的 desired_num_chunks
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


## Usage
with open(..., "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

# 用法示例
# 以下是一个串行实现，但你可以通过将每对 start/end 发送到一组进程来并行化。
    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # 对每个分块运行 pre-tokenization，并存储每个 pre-token 的计数
        # Run pre-tokenization on your chunk and store the counts for each pre-token
