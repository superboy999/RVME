#!/usr/bin/env python3

def read_numbers(filename):
    """
    从文件中读取所有有符号整数，按空格/换行分隔
    返回 int 列表
    """
    numbers = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            numbers.extend(int(x) for x in line.split())
    return numbers


def main():
    file1 = "layernorm.out"
    file2 = "layernorm_2.out"

    nums1 = read_numbers(file1)
    nums2 = read_numbers(file2)

    total1 = len(nums1)
    total2 = len(nums2)

    print(f"{file1} 数据量: {total1}")
    print(f"{file2} 数据量: {total2}")

    if total1 != total2:
        print("❌ 数据数量不一致，无法逐一比较")
        return

    mismatch_count = 0
    first_mismatch = None

    for i, (a, b) in enumerate(zip(nums1, nums2)):
        if a != b:
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = (i, a, b)
        

    print(f"总数据量: {total1}")
    print(f"不相等的数据数量: {mismatch_count}")

    if mismatch_count == 0:
        print("✅ 所有数据完全一致")
    else:
        idx, v1, v2 = first_mismatch
        print("❌ 存在不一致数据")
        print(f"第一次不一致位置: index = {idx}")
        print(f"{file1}: {v1}, {file2}: {v2}")


if __name__ == "__main__":
    main()
