import numpy as np
import cv2


def main():
    src = cv2.imread('./resources/texture.jpg')
    src = src.astype(np.float32)

    # parameters :
    randomness = 15
    divide = 4

    height, width, _ = src.shape
    bl = int(width / divide)  # block length

    result = np.zeros([3000, 3000, 3])

    init_y = np.random.randint(height - bl - 1)
    init_x = np.random.randint(width - bl - 1)

    block = np.copy(src[init_y:init_y + bl, init_x:init_x + bl, :])
    result[0:bl, 0:bl, :] = np.copy(block)

    result = result.astype(np.float32)

    def getNextBlockRow(block, src):

        strip = np.copy(block[:, int(bl / 5 * 4):, :])
        strip_h, strip_w, _ = strip.shape

        ssd = np.zeros([height, width])

        for i in range(3):
            ker = np.ones(strip[:, :, i].shape)
            C = np.sum(strip[:, :, i] * strip[:, :, i])
            ssd += cv2.filter2D(src[:, :, i] * src[:, :, i], -1, ker) - 2 * (
                cv2.filter2D(src[:, :, i], -1, strip[:, :, i])) + C

        ssd = ssd[int(strip_h / 2):height - bl, int(strip_w / 2):width - bl]

        ssd_flat = ssd.flatten()

        k = randomness

        random_index = np.argpartition(ssd_flat, k)[np.random.randint(k)]

        x = int(random_index % ssd.shape[1])
        y = int(random_index / ssd.shape[1])

        next_block = np.copy(src[y:y + bl, x:x + bl, :])

        next_strip = np.copy(next_block[0:strip_h, 0:strip_w, :])

        strip_diff = strip - next_strip
        strip_diff *= strip_diff

        strip_diff = cv2.cvtColor(strip_diff, cv2.COLOR_BGR2GRAY)

        cost = np.copy(strip_diff)
        for i in range(1, strip_h):
            for j in range(strip_w):
                if j == 0:
                    cost[i, j] += min(cost[i - 1, j], cost[i - 1, j + 1])
                elif j == strip_w - 1:
                    cost[i, j] += min(cost[i - 1, j], cost[i - 1, j - 1])
                else:
                    cost[i, j] += min(min(cost[i - 1, j], cost[i - 1, j - 1]), cost[i - 1, j + 1])

        res_strip = np.zeros(strip.shape)

        line_index = np.argmin(cost[strip_h - 1])
        for i in range(strip_h - 1, -1, -1):

            res_strip[i, :line_index, :] = strip[i, :line_index, :]
            res_strip[i, line_index:, :] = next_strip[i, line_index:, :]
            if line_index == 0:
                line_index += np.argmin((cost[i])[line_index:line_index + 2])
            elif line_index == strip_w - 1:
                line_index += np.argmin((cost[i])[line_index - 1:line_index + 1]) - 1
            else:
                line_index += np.argmin((cost[i])[line_index - 1:line_index + 2]) - 1

        next_block[:, 0:strip_w, :] = res_strip
        return next_block

    def getNextBlockCol(block, src):

        strip = np.copy(block[int(bl / 5 * 4):, :, :])
        strip_h, strip_w, _ = strip.shape

        ssd = np.zeros([height, width])

        for i in range(3):
            ker = np.ones(strip[:, :, i].shape)
            C = np.sum(strip[:, :, i] * strip[:, :, i])
            ssd += cv2.filter2D(src[:, :, i] * src[:, :, i], -1, ker) - 2 * (
                cv2.filter2D(src[:, :, i], -1, strip[:, :, i])) + C

        ssd = ssd[int(strip_h / 2):height - bl, int(strip_w / 2):width - bl]

        ssd_flat = ssd.flatten()

        k = randomness

        random_index = np.argpartition(ssd_flat, k)[np.random.randint(k)]

        x = int(random_index % ssd.shape[1])
        y = int(random_index / ssd.shape[1])

        next_block = np.copy(src[y:y + bl, x:x + bl, :])

        next_strip = np.copy(next_block[0:strip_h, 0:strip_w, :])

        strip_diff = strip - next_strip
        strip_diff *= strip_diff

        strip_diff = cv2.cvtColor(strip_diff, cv2.COLOR_BGR2GRAY)

        cost = np.copy(strip_diff)
        for j in range(1, strip_w):
            for i in range(strip_h):
                if i == 0:
                    cost[i, j] += min(cost[i, j - 1], cost[i + 1, j - 1])
                elif i == strip_h - 1:
                    cost[i, j] += min(cost[i, j - 1], cost[i - 1, j - 1])
                else:
                    cost[i, j] += min(min(cost[i, j - 1], cost[i - 1, j - 1]), cost[i + 1, j - 1])

        res_strip = np.zeros(strip.shape)

        line_index = np.argmin(cost[:, strip_w - 1])

        for j in range(strip_w - 1, -1, -1):

            res_strip[:line_index, j, :] = strip[:line_index, j, :]
            res_strip[line_index:, j, :] = next_strip[line_index:, j, :]
            if line_index == 0:
                line_index += np.argmin((cost[:, i])[line_index:line_index + 2])
            elif line_index == strip_h - 1:
                line_index += np.argmin((cost[:, j])[line_index - 1:line_index + 1]) - 1
            else:
                line_index += np.argmin((cost[:, j])[line_index - 1:line_index + 2]) - 1

        next_block[0:strip_h, :, :] = res_strip
        return next_block

    def getNextBlockMid(block, src):

        ssd = np.zeros([height, width])

        ker = np.ones([bl, bl])
        ker[int(bl / 5):, int(bl / 5):] = 0

        for i in range(3):
            C = np.sum(block[:, :, i] * block[:, :, i])
            ssd += cv2.filter2D(src[:, :, i] * src[:, :, i], -1, ker) - 2 * (
                cv2.filter2D(src[:, :, i], -1, block[:, :, i])) + C

        ssd = ssd[int(bl / 2):height - bl, int(bl / 2):width - bl]

        ssd_flat = ssd.flatten()

        k = randomness

        random_index = np.argpartition(ssd_flat, k)[np.random.randint(k)]

        x = int(random_index % ssd.shape[1])
        y = int(random_index / ssd.shape[1])

        next_block = np.copy(src[y:y + bl, x:x + bl, :])

        block_diff = block - next_block
        block_diff *= block_diff
        block_diff = cv2.cvtColor(block_diff, cv2.COLOR_BGR2GRAY)

        ver_diff = np.copy(block_diff[:, :int(bl / 5)])
        hor_diff = np.copy(block_diff[:int(bl / 5), :])

        cost = np.copy(ver_diff)
        for i in range(1, bl):
            for j in range(int(bl / 5)):
                if j == 0:
                    cost[i, j] += min(cost[i - 1, j], cost[i - 1, j + 1])
                elif j == int(bl / 5) - 1:
                    cost[i, j] += min(cost[i - 1, j], cost[i - 1, j - 1])
                else:
                    cost[i, j] += min(min(cost[i - 1, j], cost[i - 1, j - 1]), cost[i - 1, j + 1])

        res_block = np.copy(next_block)

        line_index = np.argmin(cost[bl - 1])
        for i in range(bl - 1, -1, -1):

            res_block[i, :line_index, :] = block[i, :line_index, :]

            if line_index == 0:
                line_index += np.argmin((cost[i])[line_index:line_index + 2])
            elif line_index == int(bl / 5) - 1:
                line_index += np.argmin((cost[i])[line_index - 1:line_index + 1]) - 1
            else:
                line_index += np.argmin((cost[i])[line_index - 1:line_index + 2]) - 1

        cost = np.copy(hor_diff)
        for j in range(1, bl):
            for i in range(int(bl / 5)):
                if i == 0:
                    cost[i, j] += min(cost[i, j - 1], cost[i + 1, j - 1])
                elif i == int(bl / 5) - 1:
                    cost[i, j] += min(cost[i, j - 1], cost[i - 1, j - 1])
                else:
                    cost[i, j] += min(min(cost[i, j - 1], cost[i - 1, j - 1]), cost[i + 1, j - 1])

        line_index = np.argmin(cost[:, bl - 1])

        for j in range(bl - 1, -1, -1):

            res_block[:line_index, j, :] = block[:line_index, j, :]

            if line_index == 0:
                line_index += np.argmin((cost[:, i])[line_index:line_index + 2])
            elif line_index == int(bl / 5) - 1:
                line_index += np.argmin((cost[:, j])[line_index - 1:line_index + 1]) - 1
            else:
                line_index += np.argmin((cost[:, j])[line_index - 1:line_index + 2]) - 1

        return res_block

    for i in range(1, int(np.ceil(2500 / int(bl / 5 * 4)))):
        block = getNextBlockRow(block, src)
        result[0:bl, int(bl / 5 * 4) * i:bl + int(bl / 5 * 4) * i, :] = block

    block = np.copy(result[0:bl, 0:bl, :])
    for i in range(1, int(np.ceil(2500 / int(bl / 5 * 4)))):
        block = getNextBlockCol(block, src)
        result[int(bl / 5 * 4) * i:bl + int(bl / 5 * 4) * i, 0:bl, :] = block

    for i in range(1, int(np.ceil(2500 / int(bl / 5 * 4)))):
        for j in range(1, int(np.ceil(2500 / int(bl / 5 * 4)))):
            block = np.copy(
                result[int(bl / 5 * 4) * i:bl + int(bl / 5 * 4) * i, int(bl / 5 * 4) * j:bl + int(bl / 5 * 4) * j, :])
            result[int(bl / 5 * 4) * i:bl + int(bl / 5 * 4) * i, int(bl / 5 * 4) * j:bl + int(bl / 5 * 4) * j,
            :] = getNextBlockMid(block, src)

    result = result[:2500, :2500, :]

    first_image = np.ones([2500, 2500, 3]) * 255
    first_image[1000:1000 + height, 1000:1000 + width, :] = src

    res = np.concatenate((first_image, result), axis=1)

    cv2.imwrite('./results/texture-synthesized.jpg', res)




if __name__ == "__main__":
    main()
