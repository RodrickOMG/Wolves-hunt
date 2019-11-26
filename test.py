import random
import math
import numpy as np
import matplotlib.pyplot as plt

sheep_n: int = 10  # the amount of sheep
max_size = 2000  # map size
init_center = max_size / 2  # 初始化羊群中心位置
init_distance = 100  # 初始化狼距离羊群中心的位置
sheep_x = []
sheep_y = []
sheep_v = 11.1  # the speed of sheep
wolf_v = 13.6  # the speed of wolf
sheep = []  # the dynamic coordinates list of sheep
wolf_x = []
wolf_y = []
round_count = 0  # 轮数计数器
delta_t = 0.5
d = []  # distance list between sheep and wolf

obs_n = 8
d_sheep_obs = []  # 外层表示障碍物编号，内层表示与每一只羊的距离
d_wolf_obs = []
obstacles = []  # the coordinates and radius list of obstacles

fig, ax = plt.subplots()


def init():
    # fig = plt.axis([0, max_size, 0, max_size])

    init_sheep_x = init_center  # 初始化羊群中心点x
    init_sheep_y = init_center  # 初始化羊群中心点y
    init_wolf_x = np.random.randint(init_sheep_x - init_distance, init_sheep_x + init_distance)
    init_wolf_y = np.random.randint(init_sheep_y - init_distance, init_sheep_y + init_distance)
    wolf_x.append(init_wolf_x)
    wolf_y.append(init_wolf_y)
    for i in range(sheep_n):  # 随机生成羊群
        x = np.random.randint(init_sheep_x - 5, init_sheep_x + 5)  # 在羊群中心周围随机生成第i只羊x坐标
        sheep_x.append(x)
        y = np.random.randint(init_sheep_y - 5, init_sheep_y + 5)  # 在羊群中心周围随机生成第i只羊y坐标
        sheep_y.append(y)
        sheep.append((x, y))
        d.append(calculate_d(i))
    for i in range(obs_n):  # 随机生成障碍物  # 第i个障碍物
        x = np.random.randint(init_sheep_x - 300, init_sheep_x + 300)  # 在羊群中心周围随机生成第i个障碍物x坐标
        y = np.random.randint(init_sheep_y - 300, init_sheep_y + 300)  # 在羊群中心周围随机生成第i个障碍物y坐标
        r = np.random.randint(3, 8)
        obstacles.append((x, y, r))
        temp = []
        d_sheep_obs.append(temp)
        for j in range(sheep_n):
            d_sheep_obs[i].append(calculate_d_sheep_obs(j, i))  # 初始化d_sheep_obs

    update_plt()


def calculate_d(sheep_i):
    """
    calculate the distance between sheep and wolf
    :param sheep_i: the index of sheep
    :return: distance
    """
    distance = math.sqrt((sheep_x[sheep_i] - wolf_x[0]) ** 2 + (sheep_y[sheep_i] - wolf_y[0]) ** 2)
    return distance


def calculate_d_sheep_obs(sheep_i, obs_i):
    """
    calculate the distance between sheep and obstacles
    :param sheep_i: the index of sheep
    :param obs_i: the index of obstacles
    :return: distance
    """
    distance = math.sqrt((sheep_x[sheep_i] - obstacles[obs_i][0]) ** 2
                         + (sheep_y[sheep_i] - obstacles[obs_i][1]) ** 2)
    return distance


def update_d_sheep_obs(sheep_i):
    """
    update the list of d_sheep_obs
    :param sheep_i: the index of sheep
    :return: NULL
    """
    for i in range(obs_n):
        d_sheep_obs[i][sheep_i] = calculate_d_sheep_obs(sheep_i, i)


def update_sheep():
    for i in range(sheep_n):
        sheep_x[i] = sheep_x[i] + ((sheep_x[i] - wolf_x[0]) / d[i]) * delta_t * sheep_v
        sheep_y[i] = sheep_y[i] + ((sheep_y[i] - wolf_y[0]) / d[i]) * delta_t * sheep_v
        d[i] = calculate_d(i)  # 更新狼和羊群的距离
        update_d_sheep_obs(i)


def update_d():
    for i in range(sheep_n):
        d[i] = calculate_d(i)


def find_min_distance():
    """
    find the index of minimum distance in list of d
    :return:
    """
    return np.argmin(d)


def update_wolf():
    i = find_min_distance()
    wolf_x[0] = wolf_x[0] + ((sheep_x[i] - wolf_x[0]) / d[i]) * delta_t * wolf_v
    wolf_y[0] = wolf_y[0] + ((sheep_y[i] - wolf_y[0]) / d[i]) * delta_t * wolf_v
    d[i] = calculate_d(i)
    update_d()
    print(d[i])
    if d[i] <= 0.5:
        kill(d[i])


def on_click(event):
    update_sheep()
    update_wolf()
    update_plt()


def update_plt():
    global round_count
    round_count += 1
    fig.canvas.mpl_connect('button_press_event', on_click)  # 设置鼠标按键进行下一轮操作
    plt.clf()
    plt.scatter(sheep_x, sheep_y, s=10, c='lightblue', label='sheep')  # 更新羊群坐标
    plt.scatter(wolf_x, wolf_y, s=10, c='gray', label='wolf')  # 更新狼的坐标
    plt.scatter(obstacles[0][0], obstacles[0][1], s=obstacles[0][2], c='red', label='obstacles')
    for i in range(obs_n-1):
        plt.scatter(obstacles[i+1][0], obstacles[i+1][1], s=obstacles[i+1][2] * 15, c='red')  # 更新羊群坐标
    plt.title("round " + str(round_count))
    plt.legend()
    plt.pause(9999)


def kill(distance):
    kill_rate = np.random.random()  # 随机生成一个捕杀成功率
    print("kill rate: ", kill_rate)
    if kill_rate < 0.14:  # 根据资料显示狼捕杀成功率平均在14%
        print("Successfully hunt")


if __name__ == '__main__':
    init()
    print(obstacles)
    print(d_sheep_obs)
    print(d)
    print(find_min_distance())
