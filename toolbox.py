import glob, shutil


def printbar(i, n, width=10):
    left = '=' * ((i + 1) * width // n)
    right = '.' * (width - (i + 1) * width // n)
    percent = (i + 1) / n * 100
    print('processed: {}{} {:3.0f}% '.format(left, right, percent), end='\r', flush=True)


def duplicate_json(base_dir, num):
    for jpg in glob.glob(base_dir + '/*.jpg'):
        for i in range(num):
            shutil.copy(jpg, jpg[:-4] + '_' + str(i) + '.jpg')
    for json in glob.glob(base_dir + '/*.json'):
        for i in range(num):
            shutil.copy(json, json[:-5] + '_' + str(i) + '.json')


if __name__ == '__main__':
    # printbar
    total = 10000000
    for now in range(total):
        printbar(now, total)

    # duplicate_json
    path = r'D:\2-deep_learning\data\mobianji\ori\BALI'
    duplicate_json(path, 5)
