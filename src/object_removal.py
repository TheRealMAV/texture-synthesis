import numpy as np
import cv2


def main():
    sea = cv2.imread('./resources/sea.jpg')
    sea = sea.astype(np.float32)

    jungle = cv2.imread('./resources/jungle.jpg')
    jungle = jungle.astype(np.float32)

    def getNextBlockRightDown(block, src):

        height, width, _ = src.shape
        ssd = np.zeros([height, width])
        src = src.astype(np.float32)

        ker = np.ones([bl, bl])
        ker[int(bl / 5):, int(bl / 5):] = 0
        ker = ker.astype(np.float32)

        block[int(bl / 5):, int(bl / 5):] = 0

        for i in range(3):
            C = np.sum(block[:, :, i] * block[:, :, i])
            ssd += cv2.filter2D(src[:, :, i] * src[:, :, i], -1, ker) - 2 * (
                cv2.filter2D(src[:, :, i], -1, block[:, :, i])) + C

        ssd = ssd[int(bl / 2):height - int(bl / 2), int(bl / 2):width - int(bl / 2)]

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

    def getNextBlockEdgeRight(block, src):

        height, width, _ = src.shape
        ssd = np.zeros([height, width])
        src = src.astype(np.float32)

        ker = np.ones([bl, bl])
        ker[int(bl / 5):int(4 * bl / 5), int(bl / 5):] = 0
        ker = ker.astype(np.float32)

        block[int(bl / 5):int(4 * bl / 5), int(bl / 5):] = 0

        for i in range(3):
            C = np.sum(block[:, :, i] * block[:, :, i])
            ssd += cv2.filter2D(src[:, :, i] * src[:, :, i], -1, ker) - 2 * (
                cv2.filter2D(src[:, :, i], -1, block[:, :, i])) + C

        ssd = ssd[int(bl / 2):height - int(bl / 2), int(bl / 2):width - int(bl / 2)]

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
        hor_diffUp = np.copy(block_diff[:int(bl / 5), :])
        hor_diffDown = np.copy(block_diff[int(4 * bl / 5):, :])

        res_block = np.copy(next_block)

        cost = np.copy(ver_diff)
        for i in range(1, bl):
            for j in range(int(bl / 5)):
                if j == 0:
                    cost[i, j] += min(cost[i - 1, j], cost[i - 1, j + 1])
                elif j == int(bl / 5) - 1:
                    cost[i, j] += min(cost[i - 1, j], cost[i - 1, j - 1])
                else:
                    cost[i, j] += min(min(cost[i - 1, j], cost[i - 1, j - 1]), cost[i - 1, j + 1])

        line_index = np.argmin(cost[bl - 1])
        for i in range(bl - 1, -1, -1):

            res_block[i, :line_index, :] = block[i, :line_index, :]

            if line_index == 0:
                line_index += np.argmin((cost[i])[line_index:line_index + 2])
            elif line_index == int(bl / 5) - 1:
                line_index += np.argmin((cost[i])[line_index - 1:line_index + 1]) - 1
            else:
                line_index += np.argmin((cost[i])[line_index - 1:line_index + 2]) - 1

        cost = np.copy(hor_diffDown)
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

            res_block[int(bl * 4 / 5) + line_index:, j, :] = block[int(bl * 4 / 5) + line_index:, j, :]

            if line_index == 0:
                line_index += np.argmin((cost[:, i])[line_index:line_index + 2])
            elif line_index == int(bl / 5) - 1:
                line_index += np.argmin((cost[:, j])[line_index - 1:line_index + 1]) - 1
            else:
                line_index += np.argmin((cost[:, j])[line_index - 1:line_index + 2]) - 1

        cost = np.copy(hor_diffUp)
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

            res_block[int(bl * 4 / 5) + line_index:, j, :] = block[int(bl * 4 / 5) + line_index:, j, :]

            if line_index == 0:
                line_index += np.argmin((cost[:, i])[line_index:line_index + 2])
            elif line_index == int(bl / 5) - 1:
                line_index += np.argmin((cost[:, j])[line_index - 1:line_index + 1]) - 1
            else:
                line_index += np.argmin((cost[:, j])[line_index - 1:line_index + 2]) - 1

        return res_block

    def getNextBlockEdgeDown(block, src):

        height, width, _ = src.shape
        ssd = np.zeros([height, width])
        src = src.astype(np.float32)

        ker = np.ones([bl, bl])
        ker[int(bl / 5):, int(bl / 5):int(4 * bl / 5)] = 0
        ker = ker.astype(np.float32)

        block[int(bl / 5):, int(bl / 5):int(4 * bl / 5)] = 0

        for i in range(3):
            C = np.sum(block[:, :, i] * block[:, :, i])
            ssd += cv2.filter2D(src[:, :, i] * src[:, :, i], -1, ker) - 2 * (
                cv2.filter2D(src[:, :, i], -1, block[:, :, i])) + C

        ssd = ssd[int(bl / 2):height - int(bl / 2), int(bl / 2):width - int(bl / 2)]

        ssd_flat = ssd.flatten()

        k = randomness

        random_index = np.argpartition(ssd_flat, k)[np.random.randint(k)]

        x = int(random_index % ssd.shape[1])
        y = int(random_index / ssd.shape[1])

        next_block = np.copy(src[y:y + bl, x:x + bl, :])

        block_diff = block - next_block
        block_diff *= block_diff
        block_diff = cv2.cvtColor(block_diff, cv2.COLOR_BGR2GRAY)

        ver_diffLeft = np.copy(block_diff[:, :int(bl / 5)])
        ver_diffRight = np.copy(block_diff[:, int(4 * bl / 5):])
        hor_diffUp = np.copy(block_diff[:int(bl / 5), :])

        res_block = np.copy(next_block)

        cost = np.copy(ver_diffLeft)
        for i in range(1, bl):
            for j in range(int(bl / 5)):
                if j == 0:
                    cost[i, j] += min(cost[i - 1, j], cost[i - 1, j + 1])
                elif j == int(bl / 5) - 1:
                    cost[i, j] += min(cost[i - 1, j], cost[i - 1, j - 1])
                else:
                    cost[i, j] += min(min(cost[i - 1, j], cost[i - 1, j - 1]), cost[i - 1, j + 1])

        line_index = np.argmin(cost[bl - 1])
        for i in range(bl - 1, -1, -1):

            res_block[i, :line_index, :] = block[i, :line_index, :]

            if line_index == 0:
                line_index += np.argmin((cost[i])[line_index:line_index + 2])
            elif line_index == int(bl / 5) - 1:
                line_index += np.argmin((cost[i])[line_index - 1:line_index + 1]) - 1
            else:
                line_index += np.argmin((cost[i])[line_index - 1:line_index + 2]) - 1

        cost = np.copy(ver_diffRight)
        for i in range(1, bl):
            for j in range(int(bl / 5)):
                if j == 0:
                    cost[i, j] += min(cost[i - 1, j], cost[i - 1, j + 1])
                elif j == int(bl / 5) - 1:
                    cost[i, j] += min(cost[i - 1, j], cost[i - 1, j - 1])
                else:
                    cost[i, j] += min(min(cost[i - 1, j], cost[i - 1, j - 1]), cost[i - 1, j + 1])

        line_index = np.argmin(cost[bl - 1])
        for i in range(bl - 1, -1, -1):

            res_block[i, int(bl * 4 / 5) + line_index:, :] = block[i, int(bl * 4 / 5) + line_index:, :]

            if line_index == 0:
                line_index += np.argmin((cost[i])[line_index:line_index + 2])
            elif line_index == int(bl / 5) - 1:
                line_index += np.argmin((cost[i])[line_index - 1:line_index + 1]) - 1
            else:
                line_index += np.argmin((cost[i])[line_index - 1:line_index + 2]) - 1

        cost = np.copy(hor_diffUp)
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

    def removeObject(image, init_x, init_y, patch_width, patch_height, source, bl, randomness):

        result = np.copy(image)

        for i in range(0, int(np.ceil(patch_height / int(bl / 5 * 4)))):
            for j in range(0, int(np.ceil(patch_width / int(bl / 5 * 4)))):
                y = init_y - int(bl / 5) + int(bl / 5 * 4) * i
                x = init_x - int(bl / 5) + int(bl / 5 * 4) * j
                block = np.copy(result[y: y + bl, x: x + bl, :])
                result[y: y + bl, x: x + bl, :] = getNextBlockRightDown(block, source)

        for j in range(0, int(np.ceil(patch_width / int(bl / 5 * 4)))):
            y = init_y - int(bl / 5) + int(bl / 5 * 4) * i + int(bl / 2)
            x = init_x - int(bl / 5) + int(bl / 5 * 4) * j
            block = np.copy(result[y: y + bl, x: x + bl, :])
            result[y: y + bl, x: x + bl, :] = getNextBlockEdgeRight(block, source)

        for i in range(0, int(np.ceil(patch_height / int(bl / 5 * 4)))):
            y = init_y - int(bl / 5) + int(bl / 5 * 4) * i
            x = init_x - int(bl / 5) + int(bl / 5 * 4) * j + int(bl / 2)
            block = np.copy(result[y: y + bl, x: x + bl, :])
            result[y: y + bl, x: x + bl, :] = getNextBlockEdgeDown(block, source)

        return result

    # Swimmer
    init_x, init_y = (742, 700)
    patch_width, patch_height = (213, 463)
    source = np.copy(sea[:, 960:, :])
    bl = 40
    randomness = 3
    res16 = removeObject(sea, init_x, init_y, patch_width, patch_height, source, bl, randomness)

    # Button birds
    init_x, init_y = (824, 713)
    patch_width, patch_height = (167, 216)
    source = np.copy(jungle[738:955, 361:822, :])
    bl = 20
    randomness = 3
    res15 = removeObject(jungle, init_x, init_y, patch_width, patch_height, source, bl, randomness)

    # Bird on the right
    init_x, init_y = (1131, 575)
    patch_width, patch_height = (115, 203)
    source = np.copy(jungle[380:, 1240:, :])
    bl = 15
    randomness = 3
    res15 = removeObject(res15, init_x, init_y, patch_width, patch_height, source, bl, randomness)

    # Birds on the upper left (1)
    init_x, init_y = (356, 70)
    patch_width, patch_height = (85, 96)
    source = np.copy(jungle[150:457, 467:573, :])
    bl = 10
    randomness = 1
    res15 = removeObject(res15, init_x, init_y, patch_width, patch_height, source, bl, randomness)

    # Birds on the upper left (2)
    init_x, init_y = (478, 42)
    patch_width, patch_height = (100, 84)
    source = np.copy(jungle[:64, :677, :])
    bl = 10
    randomness = 1
    res15 = removeObject(res15, init_x, init_y, patch_width, patch_height, source, bl, randomness)

    # Birds on the upper left (3)
    init_x, init_y = (440, 53)
    patch_width, patch_height = (64, 83)
    source = np.copy(jungle[:64, :677, :])
    bl = 10
    randomness = 1
    res15 = removeObject(res15, init_x, init_y, patch_width, patch_height, source, bl, randomness)

    # Birds on the upper left (4)
    init_x, init_y = (329, 110)
    patch_width, patch_height = (45, 25)
    source = np.copy(jungle[:64, :677, :])
    bl = 10
    randomness = 1
    res15 = removeObject(res15, init_x, init_y, patch_width, patch_height, source, bl, randomness)

    cv2.imwrite('./results/swimmer-removed.jpg', res16)
    cv2.imwrite('./results/birds-removed.jpg', res15)


if __name__ == "__main__":
    main()
