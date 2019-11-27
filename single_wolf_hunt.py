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
sheep_flag = []  # 是否遇到障碍物
sheep_v = 11.1  # the speed of sheep
elude_tan = []  # 进入躲避模式后设立的运动切点方向
elude_flag = []  # 进入躲避模式后是否逃脱的标志
elude_sheep = []  # 进入躲避模式后保存羊初始位置的坐标信息
predict_wolf = (0, 0)
wolf_v = 13.6  # the speed of wolf
wolf_x = []  # 狼的横坐标
wolf_y = []  # 狼的纵坐标
round_count = 0  # 轮数计数器
delta_t = 0.5  # 时间间隔
d = []  # distance list between sheep and wolf
wolf_flag: bool = False  # 狼的目标猎物是否进入躲避模式

obs_n = 10
d_sheep_obs = []  # 外层表示障碍物编号，内层表示与每一只羊的距离
d_wolf_obs = []
obstacles = []  # the coordinates and radius list of obstacles

fig, ax = plt.subplots()


def init():
    init_sheep_x = init_center  # 初始化羊群中心点x
    init_sheep_y = init_center  # 初始化羊群中心点y
    seed = np.random.randint(0, 2)
    if seed == 1:  # 分区域随机生成狼的位置
        init_wolf_x = np.random.randint(init_sheep_x - init_distance, init_sheep_x - 50)
    else:
        init_wolf_x = np.random.randint(init_sheep_x + 50, init_sheep_x + init_distance)
    seed = np.random.randint(0, 2)
    if seed == 1:
        init_wolf_y = np.random.randint(init_sheep_y - init_distance, init_sheep_y - 50)
    else:
        init_wolf_y = np.random.randint(init_sheep_y + 50, init_sheep_y + init_distance)
    wolf_x.append(init_wolf_x)
    wolf_y.append(init_wolf_y)
    for i in range(sheep_n):  # 随机生成羊群
        x = np.random.randint(init_sheep_x - 5, init_sheep_x + 5)  # 在羊群中心周围随机生成第i只羊x坐标
        sheep_x.append(x)
        y = np.random.randint(init_sheep_y - 5, init_sheep_y + 5)  # 在羊群中心周围随机生成第i只羊y坐标
        sheep_y.append(y)
        sheep_flag.append(False)
        d.append(calculate_d(x, y, wolf_x[0], wolf_y[0]))
    for i in range(obs_n):  # 随机生成障碍物  # 第i个障碍物
        seed = np.random.randint(0, 2)
        if seed == 1:
            x = np.random.randint(init_sheep_x - 500, init_sheep_x - 50)  # 在羊群中心周围随机生成第i个障碍物x坐标
        else:
            x = np.random.randint(init_sheep_x + 50, init_sheep_x + 500)
        seed = np.random.randint(0, 2)
        if seed == 1:
            y = np.random.randint(init_sheep_y - 500, init_sheep_y - 50)  # 在羊群中心周围随机生成第i个障碍物y坐标
        else:
            y = np.random.randint(init_sheep_y + 50, init_sheep_y + 500)
        r = np.random.randint(10, 30)
        obstacles.append((x, y, r))
        d_sheep_obs.append([])
        elude_tan.append([])
        elude_flag.append([])
        elude_sheep.append([])
        for j in range(sheep_n):
            d_sheep_obs[i].append(calculate_d(sheep_x[j], sheep_y[j], obstacles[i][0], obstacles[i][1]))
            elude_tan[i].append((0, 0))
            elude_flag[i].append(True)
            elude_sheep[i].append((0, 0))
            # 初始化躲避模式相关参数

    update_plt()


def calculate_d(x1, y1, x2, y2):
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance


def update_d_sheep_obs(sheep_i):
    """
    update the list of d_sheep_obs
    :param sheep_i: the index of sheep
    :return: NULL
    """
    for i in range(obs_n):
        d_sheep_obs[i][sheep_i] = calculate_d(sheep_x[sheep_i], sheep_y[sheep_i], obstacles[i][0], obstacles[i][1])


def update_sheep_normal(i):
    sheep_x[i] = sheep_x[i] + ((sheep_x[i] - wolf_x[0]) / d[i]) * delta_t * sheep_v
    sheep_y[i] = sheep_y[i] + ((sheep_y[i] - wolf_y[0]) / d[i]) * delta_t * sheep_v
    d[i] = calculate_d(sheep_x[i], sheep_y[i], wolf_x[0], wolf_y[0])  # 更新狼和羊群的距离
    update_d_sheep_obs(i)


def update_sheep():
    for i in range(sheep_n):
        for j in range(obs_n):
            if d_sheep_obs[j][i] <= 2.5 * obstacles[j][2]:
                sheep_flag[i] = True
                if elude_flag[j][i]:
                    elude_obstacles(i, j)
                    break
                else:
                    update_sheep_obs(elude_tan[j][i][0], elude_tan[j][i][1], i, j)
                    break
            else:
                sheep_flag[i] = False
        if not sheep_flag[i]:
            update_sheep_normal(i)


def elude_obstacles(sheep_i, k):
    """
    if the distance between sheep_i and obs_i less than 2.5, it should elude this obstacle
    :param sheep_i: the index of sheep_i
    :param k:  the index of obstacles_i
    :return:
    """
    # Ax+By+C=0
    global predict_wolf
    r = obstacles[k][2]
    xk = obstacles[k][0]
    yk = obstacles[k][1]
    a = (sheep_y[sheep_i] - wolf_y[0]) / (sheep_x[sheep_i] - wolf_x[0])
    b = -1
    c = sheep_y[sheep_i] - (a * sheep_x[sheep_i])
    d_obs_dir = math.fabs(a * xk + b * yk + c) / math.sqrt(a ** 2 + b ** 2)
    if d_obs_dir < r:  # 羊运动方向会撞上障碍物
        sheep_flag[sheep_i] = True
        a2 = -1 / (xk - sheep_x[sheep_i])
        b2 = yk - sheep_y[sheep_i]
        n1 = (-a2 ** 2 * b2 * r ** 2 + r * math.sqrt(a2 ** 2 * b2 ** 2 + 1 - a2 ** 2 * r ** 2)) / (
                a2 ** 2 * b2 ** 2 + 1)
        n2 = (-a2 ** 2 * b2 * r ** 2 - r * math.sqrt(a2 ** 2 * b2 ** 2 + 1 - a2 ** 2 * r ** 2)) / (
                a2 ** 2 * b2 ** 2 + 1)
        m1 = a2 * r ** 2 + a2 * b2 * n1
        m2 = a2 * r ** 2 + a2 * b2 * n2
        x1 = m1 + xk
        y1 = n1 + yk
        x2 = m2 + xk
        y2 = n2 + yk
        sheep_motion_list = compare_angle(x1, y1, x2, y2, sheep_i)
        x = sheep_motion_list[0]
        y = sheep_motion_list[1]
        elude_tan[k][sheep_i] = (x, y)
        elude_sheep[k][sheep_i] = (sheep_x[sheep_i], sheep_y[sheep_i])
        update_sheep_obs(x, y, sheep_i, k)  # 更新羊和障碍物距离
        target = find_min_distance()
        if sheep_i == target:
            predict_wolf = (wolf_x[0], wolf_y[0])
            update_wolf_predict(x, y, sheep_i, k)
        elude_flag[k][sheep_i] = False
    else:
        sheep_flag[sheep_i] = False
        elude_flag[k][sheep_i] = True
        update_sheep_normal(sheep_i)


def update_wolf_predict(x_tan, y_tan, i, k):
    """
    when sheep i encounters an obstacle, the wolf will make prediction
    """
    d_temp = math.sqrt((2 * x_tan - elude_sheep[k][i][0] - predict_wolf[0]) ** 2 + (2 * y_tan - elude_sheep[k][i][1]
                                                                                    - predict_wolf[1]) ** 2)
    wolf_x[0] = wolf_x[0] + (2 * x_tan - elude_sheep[k][i][0] - predict_wolf[0]) / d_temp * delta_t * wolf_v
    wolf_y[0] = wolf_y[0] + (2 * y_tan - elude_sheep[k][i][1] - predict_wolf[1]) / d_temp * delta_t * wolf_v
    d[i] = calculate_d(sheep_x[i], sheep_y[i], wolf_x[0], wolf_y[0])
    update_d_wolf()
    if d[i] <= 0.6:
        kill(d[i])


def update_sheep_obs(x_tan, y_tan, i, k):
    """
    update position of sheep who is eluding an obstacle
    :param x_tan: the x coordinate of escape route line and obstacle's tangent point
    :param y_tan: the y coordinate of escape route line and obstacle's tangent point
    :param i: the index of sheep
    :param k: the index of obs
    :return:
    """
    d_tan_sheep = calculate_d(elude_sheep[k][i][0], elude_sheep[k][i][1], x_tan, y_tan)
    sheep_x[i] = sheep_x[i] + ((x_tan - elude_sheep[k][i][0]) / d_tan_sheep) * delta_t * sheep_v * 0.8
    sheep_y[i] = sheep_y[i] + ((y_tan - elude_sheep[k][i][1]) / d_tan_sheep) * delta_t * sheep_v * 0.8
    d[i] = calculate_d(sheep_x[i], sheep_y[i], wolf_x[0], wolf_y[0])  # 更新狼和羊群的距离
    update_d_sheep_obs(i)
    if d_sheep_obs[k][i] >= 2.5 * obstacles[k][2]:
        elude_flag[k][i] = True
        sheep_flag[i] = False


def compare_angle(x1, y1, x2, y2, sheep_i):
    theta1 = calculate_acos(x1, y1, sheep_i)
    theta2 = calculate_acos(x2, y2, sheep_i)
    list1 = [x1, y1, theta1]
    list2 = [x2, y2, theta2]
    if theta1 >= theta2:
        return list1
    else:
        return list2


def calculate_acos(x, y, sheep_i):
    x_si = sheep_x[sheep_i]
    y_si = sheep_y[sheep_i]
    return math.acos(((x_si - wolf_x[0]) * (x - x_si) + (y_si - wolf_y[0]) * (y - y_si)) /
                     (math.sqrt((x_si - wolf_x[0]) ** 2 + (y_si - wolf_y[0]) ** 2) * math.sqrt(
                         (x - x_si) ** 2 + (y - y_si) ** 2)))


def update_d_wolf():
    for i in range(sheep_n):
        d[i] = calculate_d(sheep_x[i], sheep_y[i], wolf_x[0], wolf_y[0])


def find_min_distance():
    """
    find the index of minimum distance in list of d
    :return:
    """
    return np.argmin(d)


def update_wolf():
    i = find_min_distance()
    if not sheep_flag[i]:
        wolf_x[0] = wolf_x[0] + ((sheep_x[i] - wolf_x[0]) / d[i]) * delta_t * wolf_v
        wolf_y[0] = wolf_y[0] + ((sheep_y[i] - wolf_y[0]) / d[i]) * delta_t * wolf_v
        d[i] = calculate_d(sheep_x[i], sheep_y[i], wolf_x[0], wolf_y[0])
        update_d_wolf()
        if d[i] <= 0.6:
            kill(d[i])


def on_click(event):
    global wolf_flag
    update_sheep()
    target = find_min_distance()
    for i in range(obs_n):
        if not elude_flag[i][target]:
            update_wolf_predict(elude_tan[i][target][0], elude_tan[i][target][1], target, i)
            wolf_flag = True
            break
    if not wolf_flag:
        update_wolf()
    else:
        wolf_flag = False
    update_plt()


def update_plt():
    global round_count
    round_count += 1
    fig.canvas.mpl_connect('key_press_event', on_click)  # 设置鼠标按键进行下一轮操作
    plt.clf()
    plt.scatter(sheep_x, sheep_y, s=10, c='lightblue', label='sheep')  # 更新羊群坐标
    plt.scatter(wolf_x, wolf_y, s=10, c='gray', label='wolf')  # 更新狼的坐标
    plt.scatter(obstacles[0][0], obstacles[0][1], s=obstacles[0][2] * 10, c='red', label='obstacles')
    for i in range(obs_n - 1):
        plt.scatter(obstacles[i + 1][0], obstacles[i + 1][1], s=obstacles[i + 1][2] ** 2 * np.pi * 0.1,
                    c='red')  # 更新羊群坐标
    plt.title("round " + str(round_count))
    plt.legend()
    plt.pause(120)


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
