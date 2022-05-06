import torch

def construct_table(map=False):

    # eq0, ent1, rent2, neg3, alt4, cov5, ind6
    table = torch.zeros(size=(7, 7, 7), dtype=torch.float32)

    # eq + x -> x
    for i in range(7):
        table[0, i, i] = 1

    # ent + x1 -> x2
    table[1, 0, 1] = 1
    table[1, 1, 1] = 1
    table[1, 2, 6] = 1
    table[1, 3, 4] = 1
    table[1, 4, 4] = 1
    table[1, 5, 6] = 1
    table[1, 6, 6] = 1

    # rent + x1 -> x2
    table[2, 0, 2] = 1
    table[2, 1, 6] = 1
    table[2, 2, 2] = 1
    table[2, 3, 5] = 1
    table[2, 4, 4] = 1
    table[2, 5, 5] = 1
    table[2, 6, 6] = 1

    # neg + x1 -> x2
    table[3, 0, 3] = 1
    table[3, 1, 5] = 1
    table[3, 2, 4] = 1
    table[3, 3, 0] = 1
    table[3, 4, 2] = 1
    table[3, 5, 1] = 1
    table[3, 6, 6] = 1

    # alt + x1 -> x2
    table[4, 0, 4] = 1
    table[4, 1, 4] = 1
    table[4, 2, 4] = 1
    table[4, 3, 1] = 1
    table[4, 4, 4] = 1
    table[4, 5, 1] = 1
    table[4, 6, 6] = 1

    # cov + x1 -> x2
    table[5, 0, 5] = 1
    table[5, 1, 5] = 1
    table[5, 2, 6] = 1
    table[5, 3, 2] = 1
    table[5, 4, 2] = 1
    table[5, 5, 6] = 1
    table[5, 6, 6] = 1

    # ind + ? = ind
    for i in range(7):
        table[6, i, 6] = 1

    if not map:
        return table
    else:
        return torch.argmax(table, dim=-1).type(torch.int64)



if __name__ == '__main__':
    table = construct_table()
    map = torch.argmax(table, dim=-1)
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([2, 5, 4])
    print(map)
    print(map[a, b])
    print(map[0, 0] in [1, 2])
