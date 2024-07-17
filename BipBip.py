import numpy as np
from os import urandom


def int_to_bin_24(plain):
    n = plain.shape[0]
    p0 = np.zeros((n, 24), dtype=np.uint8)
    for i in range(24):
        off = 23 - i
        p0[:, i] = (plain >> off) & 1
    return p0


def int_to_bin_256(key):
    n = key[0].shape[0]
    key0 = np.zeros((n, 256), dtype=np.uint8)
    t = 0
    for i in range(256):
        j = int(i / 64)
        if j == 0:
            t = 63
        elif j == 1:
            t = 127
        elif j == 2:
            t = 191
        elif j == 3:
            t = 255
        off = t - i
        key0[:, i] = (key[j] >> off) & 1
    return key0


def int_to_bin_40(tweak):
    n = tweak.shape[0]
    t0 = np.zeros((n, 40), dtype=np.uint8)
    for i in range(40):
        off = 39 - i
        t0[:, i] = (tweak >> off) & 1
    return t0


def key_schedule(key):
    subkeys = np.zeros((7, key.shape[0], 53), dtype=np.uint8)
    k0_list = [255 - ((3 ** i) % 256) for i in range(1, 25)]
    for i in range(24):
        subkeys[0, :, i] = key[:, k0_list[i]]
    for i in range(1, 7):
        for j in range(53):
            subkeys[i, :, j] = key[:, 255 - ((53 * i) + j) % 256]
    return subkeys


def X(input):
    # 这里为了方便改变了高低位，末尾得到的结果也要改变回来
    input = np.flip(input, axis=1)
    output = input * 0
    for i in range(53):
        a1 = input[:, i % 53]
        a2 = input[:, (i + 1) % 53] ^ 1
        a3 = input[:, (i + 2) % 53]
        output[:, i] = a1 ^ (a2 * a3)
    return np.flip(output, axis=1)


def pi_4(input):
    # 这里进来的input是高低位倒置的
    output = input * 0
    for i in range(53):
        output[:, i] = input[:, (13 * i) % 53]
    return output


def pi_5(input):
    # 这里进来的input是高低位倒置的
    output = input * 0
    for i in range(53):
        output[:, i] = input[:, (11 * i) % 53]
    return output


def theta_t(input):
    # 这里进来的input是高低位倒置的
    output = input * 0
    for i in range(53):
        output[:, i] = input[:, i] ^ input[:, (i + 1) % 53] ^ input[:, (i + 8) % 53]
    return output


def theta_t_apostrophe(input):
    # 这里进来的input是高低位倒置的
    output = input * 0
    for i in range(52):
        output[:, i] = input[:, i] ^ input[:, (i + 1) % 53]
    output[:, 52] = input[:, 52]
    return output


def G(input):
    output = X(input)
    # 这里为了方便改变了高低位，末尾得到的结果也要改变回来
    # 应该X（input）里面有一个翻转了
    output = np.flip(output, axis=1)
    output = pi_5(output)
    output = theta_t(output)
    output = pi_4(output)
    return np.flip(output, axis=1)


def G_apostrophe(input):
    output = X(input)
    # 这里为了方便改变了高低位，末尾得到的结果也要改变回来
    # 应该X（input）里面有一个翻转了
    output = np.flip(output, axis=1)
    output = pi_5(output)
    output = theta_t_apostrophe(output)
    output = pi_4(output)
    return np.flip(output, axis=1)


def E(input):
    k0 = np.zeros((input.shape[0], 24), dtype=np.uint8)
    k1 = np.zeros((input.shape[0], 24), dtype=np.uint8)
    input = np.flip(input, axis=1)
    for i in range(24):
        k0[:, i] = input[:, i * 2]
        k1[:, i] = input[:, i * 2 + 1]
    return k0, k1


def E0(input):
    k0 = np.zeros((input.shape[0], 24), dtype=np.uint8)
    input = np.flip(input, axis=1)
    for i in range(24):
        k0[:, i] = input[:, i * 2]
    return k0


def tweak_schedule(tweak, subkeys):
    t = np.zeros((tweak.shape[0], 53), dtype=np.uint8)
    ks = np.zeros((12, tweak.shape[0], 24), dtype=np.uint8)

    t[:, 0:40] = tweak
    t[:, 40] = 1
    ks[0] = subkeys[0, :, 0:24]

    # get k1 k2
    output = t ^ subkeys[1]
    output = X(output)
    ks[1], ks[2] = E(output)

    # get k3 k4
    output = output ^ subkeys[2]
    output = G(output)
    ks[3], ks[4] = E(output)
    # get k5
    output = output ^ subkeys[3]
    output = G(output)
    output = G_apostrophe(output)
    ks[5] = E0(output)
    # get k6
    output = output ^ subkeys[4]
    output = G(output)
    ks[6] = E0(output)
    # get k7
    output = G_apostrophe(output)
    ks[7] = E0(output)
    # get k8
    output = output ^ subkeys[5]
    output = G(output)
    ks[8] = E0(output)
    # get k9
    output = G_apostrophe(output)
    ks[9] = E0(output)
    # get k10,k11
    output = output ^ subkeys[6]
    output = G(output)
    ks[10], ks[11] = E(output)

    return ks


def S(input):
    output = input * 0
    S_box = np.array(
        [0, 1, 2, 3, 4, 6, 62, 60, 8, 17, 14, 23, 43, 51, 53, 45, 25, 28, 9, 12, 21, 19, 61, 59, 49, 44, 37, 56, 58, 38,
         54, 42, 52, 29, 55, 30, 48, 26, 11, 33, 46, 31, 41, 24, 15, 63, 16, 32, 40, 5, 57, 20, 36, 10, 13, 35, 18, 39,
         7, 50, 27, 47, 22, 34])
    for i in range(4):
        temp = (2 ** 5) * input[:, i * 6 + 0] + (2 ** 4) * input[:, i * 6 + 1] + (2 ** 3) * input[:, i * 6 + 2] + (
                2 ** 2) * input[:, i * 6 + 3] + (2 ** 1) * input[:, i * 6 + 4] + input[:, i * 6 + 5]
        out_sbox = S_box[temp]
        for j in range(6):
            output[:, i * 6 + j] = (out_sbox >> (5 - j)) & 1
    return output


def S_inverse(input):
    output = input * 0
    # 定义逆S盒
    S_box_inverse = np.array([None] * 64)  # 初始化一个长度为64的空数组
    # 原S盒
    S_box = np.array(
        [0, 1, 2, 3, 4, 6, 62, 60, 8, 17, 14, 23, 43, 51, 53, 45, 25, 28, 9, 12, 21, 19, 61, 59, 49, 44, 37, 56, 58, 38,
         54, 42, 52, 29, 55, 30, 48, 26, 11, 33, 46, 31, 41, 24, 15, 63, 16, 32, 40, 5, 57, 20, 36, 10, 13, 35, 18, 39,
         7, 50, 27, 47, 22, 34])
    # 构建逆S盒
    for i in range(64):
        S_box_inverse[S_box[i]] = i
    for i in range(4):
        temp = (2 ** 5) * input[:, i * 6 + 0] + (2 ** 4) * input[:, i * 6 + 1] + (2 ** 3) * input[:, i * 6 + 2] + (
                2 ** 2) * input[:, i * 6 + 3] + (2 ** 1) * input[:, i * 6 + 4] + input[:, i * 6 + 5]
        out_sbox = S_box_inverse[temp]
        for j in range(6):
            output[:, i * 6 + j] = (out_sbox >> (5 - j)) & 1
    return output


def pi_1(input):
    input = np.flip(input, axis=1)
    output = input * 0
    P1 = [1, 7, 6, 0, 2, 8, 12, 18, 19, 13, 14, 20, 21, 15, 16, 22, 23, 17, 9, 3, 4, 10, 11, 5]
    for i in range(24):
        output[:, i] = input[:, P1[i]]
    return np.flip(output, axis=1)


def pi_1_inverse(input):
    input = np.flip(input, axis=1)
    output = input * 0
    # 原函数中的P1排列
    P1 = [1, 7, 6, 0, 2, 8, 12, 18, 19, 13, 14, 20, 21, 15, 16, 22, 23, 17, 9, 3, 4, 10, 11, 5]
    # 构建P1的逆排列
    P1_inverse = [None] * 24
    for i, p in enumerate(P1):
        P1_inverse[p] = i
    # 使用P1的逆排列进行元素映射
    for i in range(24):
        output[:, i] = input[:, P1_inverse[i]]
    return np.flip(output, axis=1)


def pi_2(input):
    input = np.flip(input, axis=1)
    output = input * 0
    P2 = [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11, 16, 12, 13, 17, 20, 21, 15, 14, 18, 19, 22, 23]
    for i in range(24):
        output[:, i] = input[:, P2[i]]
    return np.flip(output, axis=1)


def pi_2_inverse(input):
    input = np.flip(input, axis=1)
    output = input * 0
    # 原始排列
    P2 = [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11, 16, 12, 13, 17, 20, 21, 15, 14, 18, 19, 22, 23]
    # 构建逆排列
    P2_inverse = [0] * 24
    for i in range(24):
        P2_inverse[P2[i]] = i
    # 使用逆排列进行映射
    for i in range(24):
        output[:, i] = input[:, P2_inverse[i]]
    return np.flip(output, axis=1)


def pi_3(input):
    input = np.flip(input, axis=1)
    output = input * 0
    P3 = [16, 22, 11, 5, 2, 8, 0, 6, 19, 13, 12, 18, 14, 15, 1, 7, 21, 20, 4, 3, 17, 23, 10, 9]
    for i in range(24):
        output[:, i] = input[:, P3[i]]
    return np.flip(output, axis=1)


def theta_d(input):
    input = np.flip(input, axis=1)
    output = input * 0
    for i in range(24):
        output[:, i] = input[:, i] ^ input[:, (i + 2) % 24] ^ input[:, (i + 12) % 24]
    return np.flip(output, axis=1)


def R_apostrophe(input):
    output = pi_3(input)
    output = S(output)
    return output


def R(input):
    output = pi_2(input)
    output = theta_d(output)
    output = pi_1(output)
    output = S(output)
    return output


def encrypt(input, key):
    output = input
    output = output ^ key[1]

    for i in range(1, 4):
        output = R_apostrophe(output)
        output = output ^ key[i]
    for i in range(4, 5):
        output = R(output)
        output = output ^ key[i]
    # output1 = output
    # for i in range(5, 6):
    #     output = R(output)
    #     output = output ^ key[i]
    # for i in range(8, 11):
    #     output = output ^ key[i]
    #     output = R_apostrophe(output)
    # output = output ^ key[11]
    # return output, output1
    return output




# plain0 = np.frombuffer(urandom(4 * 1), dtype=np.uint32) & 0xffffff
# keys = np.frombuffer(urandom(8 * 4 * 1), dtype=np.uint64).reshape(4, -1)
# tweak = (np.frombuffer(urandom(8 * 1), dtype=np.uint64) & 0xffffffffff)
# p0 = int_to_bin_24(plain0)
# k0 = int_to_bin_256(keys)
# t0 = int_to_bin_40(tweak)
# subkeys = key_schedule(k0)
# subkeys_24 = tweak_schedule(t0, subkeys)
# plain0 = encrypt(p0, subkeys_24)


def make_target_diff_samples(n, diff_type=1, diff=0x8000):
    keys = np.frombuffer(urandom(8 * 4 * n), dtype=np.uint64).reshape(4, -1)
    # keys_diff = np.vstack([keys[0] ^ 0x80,keys[1:4]]).reshape(4, -1)

    tweak = np.frombuffer(urandom(8 * n), dtype=np.uint64) & 0xffffffffff
    k0 = int_to_bin_256(keys)
    # k0_diff = int_to_bin_256(keys_diff)
    t0 = int_to_bin_40(tweak)
    key = key_schedule(k0)
    # key_diff = key_schedule(k0_diff)
    ks = tweak_schedule(t0, key)
    # ks_diff = tweak_schedule(t0, key_diff)
    plain0 = np.frombuffer(urandom(4 * n), dtype=np.uint32) & 0xffffff
    if diff_type == 1:
        plain1 = plain0 ^ diff
    else:
        plain1 = np.frombuffer(urandom(4 * n), dtype=np.uint32) & 0xffffff
    p0 = int_to_bin_24(plain0)
    p1 = int_to_bin_24(plain1)

    cipher0, cipher40 = encrypt(p0, ks)
    cipher1, cipher41 = encrypt(p1, ks)

    # np.save("./save/"+str(n)+"diff"+str(diff_type)+"_4+3cipher0.npy", cipher0)
    # np.save("./save/"+str(n)+"diff"+str(diff_type)+"_4+3cipher40.npy", cipher40)
    # np.save("./save/"+str(n)+"diff"+str(diff_type)+"_4+3cipher1.npy", cipher1)
    # np.save("./save/"+str(n)+"diff"+str(diff_type)+"_4+3cipher41.npy", cipher41)

    # cipher0 = np.load("./save/" + str(n) + "diff" + str(diff_type) + "_4+3cipher0.npy")
    # cipher40 = np.load("./save/" + str(n) + "diff" + str(diff_type) + "_4+3cipher40.npy")
    # cipher1 = np.load("./save/" + str(n) + "diff" + str(diff_type) + "_4+3cipher1.npy")
    # cipher41 = np.load("./save/" + str(n) + "diff" + str(diff_type) + "_4+3cipher41.npy")
    #
    # cipher0 = np.load("./save/" + str(n) + "diff" + str(diff_type) + "cipher0.npy")
    # cipher40 = np.load("./save/" + str(n) + "diff" + str(diff_type) + "cipher40.npy")
    # cipher1 = np.load("./save/" + str(n) + "diff" + str(diff_type) + "cipher1.npy")
    # cipher41 = np.load("./save/" + str(n) + "diff" + str(diff_type) + "cipher41.npy")

    # d = np.full(n, 0x22802, dtype=np.uint32)  # high R4  4bit
    # d = np.full(n, 0x400244, dtype=np.uint32)  # low R4  4bit
    # d = np.full(n, 0xba882, dtype=np.uint32)  # high R4  8bit
    # d = np.full(n, 0x740254, dtype=np.uint32)  # low R4  8bit
    # d = np.full(n, 0x8bfdab, dtype=np.uint32)  # high R4  16bit
    # d = np.full(n, 0xf4577d, dtype=np.uint32)  # low R4  16bit
    # d = np.full(n, 0xbffdbb, dtype=np.uint32)  # high R4  20bit
    # d = np.full(n, 0xfdd7fd, dtype=np.uint32)  # low R4  20bit

    # d = np.full(n, 0x8a2, dtype=np.uint32)  # high R5  4bit
    # d = np.full(n, 0x648000, dtype=np.uint32)  # low R5  4bit
    # d = np.full(n, 0x8bf, dtype=np.uint32)  # high R5  8bit
    # d = np.full(n, 0x7cd000, dtype=np.uint32)  # low R5 8bit
    # d = np.full(n, 0x832fff, dtype=np.uint32)  # high R5  16bit
    # d = np.full(n, 0xfff740, dtype=np.uint32)  # low R5  16bit
    # d = np.full(n, 0x9b7fff, dtype=np.uint32)  # high R5  20bit
    # d = np.full(n, 0xfff75d, dtype=np.uint32)  # low R5  20bit
    d = np.full(n, 0xffffff, dtype=np.uint32)
    d = int_to_bin_24(d)
    x = np.concatenate((cipher0, cipher1, (cipher40 ^ cipher41) & d), axis=1)
    # x = np.concatenate((cipher0, cipher1, (cipher30 ^ cipher31)), axis=1)
    # x = np.concatenate((cipher0, cipher1), axis=1)
    return x



def make_dataset_with_group_size(n, diff=0x8000, group_size=2):
    assert n % group_size == 0
    num = n // 2
    X_p = make_target_diff_samples3(n=num, diff_type=1, diff=diff)
    X_n = make_target_diff_samples3(n=num, diff_type=0)
    X_raw = np.concatenate((X_p, X_n), axis=0)
    # np.save('./Data/R5needR4_10_6_8_diff_data.npy', X_raw)

    # if n == (10 ** 6) * 8:
    #     X_raw = np.load('Data/R5needR4_10_6_8_diff_data.npy')
    # else:
    #     X_raw = np.load('./Data/R5needR4_10_5_8_diff_data.npy')

    n, m = np.shape(X_raw)
    X = X_raw.reshape(-1, group_size * m)
    Y_p = [1 for i in range(num // group_size)]
    Y_n = [0 for i in range(num // group_size)]
    Y = np.concatenate((Y_p, Y_n))
    return X, Y

# X, Y = make_dataset_with_group_size(2)
