import random
import math
import numpy as np
import matplotlib.pyplot as plt

bison_n: int = 10  # the amount of bison
max_size = 2000  # map size
init_center = max_size / 2  # 初始化羊群中心位置
init_distance = 100  # 初始化狼距离羊群中心的位置
bison_x = []
bison_y = []
bison_flag = []  # 是否遇到障碍物
bison_v = 11.1  # the speed of sheep
bison_angle = []
elude_tan = []  # 进入躲避模式后设立的运动切点方向
elude_flag = []  # 进入躲避模式后是否逃脱的标志
elude_bison = []  # 进入躲避模式后保存羊初始位置的坐标信息
predict_wolf = (0, 0)
wolf_v = 13.6  # the speed of wolf
wolf_x = []  # 狼的横坐标
wolf_y = []  # 狼的纵坐标
round_count = 0  # 轮数计数器
delta_t = 0.5  # 时间间隔
d = []  # distance list between sheep and wolf
wolf_flag: bool = False  # 狼的目标猎物是否进入躲避模式

obs_n = 10
d_bison_obs = []  # 外层表示障碍物编号，内层表示与每一只羊的距离
d_wolf_obs = []
obstacles = []  # the coordinates and radius list of obstacles

fig, ax = plt.subplots()


def init():
    global d_bison_obs, elude_tan, elude_flag, elude_bison
    init_bison_x = init_center  # 初始化羊群中心点x
    init_bison_y = init_center  # 初始化羊群中心点y
    seed = np.random.randint(0, 2)
    if seed == 1:  # 分区域随机生成狼的位置
        init_wolf_x = np.random.randint(init_bison_x - init_distance, init_bison_x - 50)
    else:
        init_wolf_x = np.random.randint(init_bison_x + 50, init_bison_x + init_distance)
    seed = np.random.randint(0, 2)
    if seed == 1:
        init_wolf_y = np.random.randint(init_bison_y - init_distance, init_bison_y - 50)
    else:
        init_wolf_y = np.random.randint(init_bison_y + 50, init_bison_y + init_distance)
    wolf_x.append(init_wolf_x)
    wolf_y.append(init_wolf_y)
    for i in range(bison_n):  # 随机生成羊群
        x = np.random.randint(init_bison_x - 5, init_bison_x + 5)  # 在羊群中心周围随机生成第i只羊x坐标
        bison_x.append(x)
        y = np.random.randint(init_bison_y - 5, init_bison_y + 5)  # 在羊群中心周围随机生成第i只羊y坐标
        bison_y.append(y)
        bison_flag.append(False)
        d.append(calculate_d(x, y, wolf_x[0], wolf_y[0]))
        bison_angle.append(0)
    d_bison_obs = [[]] * obs_n
    elude_tan = [[]] * obs_n
    elude_flag = [[]] * obs_n
    elude_bison = [[]] * obs_n
    for i in range(obs_n):  # 随机生成障碍物  # 第i个障碍物
        seed = np.random.randint(0, 2)
        if seed == 1:
            x = np.random.randint(init_bison_x - 500, init_bison_x - 50)  # 在羊群中心周围随机生成第i个障碍物x坐标
        else:
            x = np.random.randint(init_bison_x + 50, init_bison_x + 500)
        seed = np.random.randint(0, 2)
        if seed == 1:
            y = np.random.randint(init_bison_y - 500, init_bison_y - 50)  # 在羊群中心周围随机生成第i个障碍物y坐标
        else:
            y = np.random.randint(init_bison_y + 50, init_bison_y + 500)
        r = np.random.randint(10, 30)
        obstacles.append((x, y, r))
        elude_tan[i] = [(0, 0)] * bison_n
        elude_flag[i] = [True] * bison_n
        elude_bison[i] = [(0, 0)] * bison_n
        for j in range(bison_n):
            d_bison_obs[i].append(calculate_d(bison_x[j], bison_y[j], obstacles[i][0], obstacles[i][1]))
            # 初始化躲避模式相关参数

    update_plt()


def calculate_d(x1, y1, x2, y2):
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance


def update_d_bison_obs(bison_i):
    """
    update the list of d_bison_obs
    :param bison_i: the index of bison
    :return: NULL
    """
    for i in range(obs_n):
        d_bison_obs[i][bison_i] = calculate_d(bison_x[bison_i], bison_y[bison_i], obstacles[i][0], obstacles[i][1])


def update_bison_normal(i):
    bison_x[i] = bison_x[i] + ((bison_x[i] - wolf_x[0]) / d[i]) * delta_t * bison_v
    bison_y[i] = bison_y[i] + ((bison_y[i] - wolf_y[0]) / d[i]) * delta_t * bison_v
    d[i] = calculate_d(bison_x[i], bison_y[i], wolf_x[0], wolf_y[0])  # 更新狼和羊群的距离
    update_d_bison_obs(i)


def update_bison():
    for i in range(bison_n):
        obs_flag = 0  # the flag of whether the bison encounter an obstacle
        for j in range(obs_n):
            if d_bison_obs[j][i] <= 2.5 * obstacles[j][2]:
                bison_flag[i] = True
                if elude_flag[j][i]:
                    elude_obstacles(i, j)
                    obs_flag = 1
                    break
                else:
                    update_bison_obs(elude_tan[j][i][0], elude_tan[j][i][1], i, j)
                    obs_flag = 1
                    break
            else:
                bison_flag[i] = False
        if not bison_flag[i] and obs_flag == 0:
            update_bison_normal(i)


def elude_obstacles(bison_i, k):
    """
    if the distance between bison_i and obs_i less than 2.5, it should elude this obstacle
    :param bison_i: the index of bison_i
    :param k:  the index of obstacles_i
    :return:
    """
    # Ax+By+C=0
    global predict_wolf
    r = obstacles[k][2]
    xk = obstacles[k][0]
    yk = obstacles[k][1]
    a = (bison_y[bison_i] - wolf_y[0]) / (bison_x[bison_i] - wolf_x[0])
    b = -1
    c = bison_y[bison_i] - (a * bison_x[bison_i])
    d_obs_dir = math.fabs(a * xk + b * yk + c) / math.sqrt(a ** 2 + b ** 2)
    if d_obs_dir < r:  # 羊运动方向会撞上障碍物
        bison_flag[bison_i] = True
        a2 = -1 / (xk - bison_x[bison_i])
        b2 = yk - bison_y[bison_i]
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
        bison_motion_list = compare_angle(x1, y1, x2, y2, bison_i)
        x = bison_motion_list[0]
        y = bison_motion_list[1]
        bison_angle[bison_i] = bison_motion_list[2]
        elude_tan[k][bison_i] = (x, y)
        elude_bison[k][bison_i] = (bison_x[bison_i], bison_y[bison_i])
        update_bison_obs(x, y, bison_i, k)  # 更新羊和障碍物距离
        target = find_min_distance()
        if bison_i == target:
            predict_wolf = (wolf_x[0], wolf_y[0])
            update_wolf_predict(x, y, bison_i, k)
        elude_flag[k][bison_i] = False
    else:
        bison_flag[bison_i] = False
        elude_flag[k][bison_i] = True
        update_bison_normal(bison_i)


def update_wolf_predict(x_tan, y_tan, i, k):
    """
    when sheep i encounters an obstacle, the wolf will make prediction
    """
    d_temp = math.sqrt((2 * x_tan - elude_bison[k][i][0] - predict_wolf[0]) ** 2 + (2 * y_tan - elude_bison[k][i][1]
                                                                                    - predict_wolf[1]) ** 2)
    turn_v = math.sqrt(wolf_v ** 2 - 10 * wolf_v * (1 - math.cos(bison_angle[i]) ** 2) / math.cos(bison_angle[i]))
    wolf_x[0] = wolf_x[0] + (2 * x_tan - elude_bison[k][i][0] - predict_wolf[0]) / d_temp * delta_t * turn_v
    wolf_y[0] = wolf_y[0] + (2 * y_tan - elude_bison[k][i][1] - predict_wolf[1]) / d_temp * delta_t * turn_v
    d[i] = calculate_d(bison_x[i], bison_y[i], wolf_x[0], wolf_y[0])
    update_d_wolf()
    if d[i] <= 0.6:
        kill(d[i])


def update_bison_obs(x_tan, y_tan, i, k):
    """
    update position of sheep who is eluding an obstacle
    :param x_tan: the x coordinate of escape route line and obstacle's tangent point
    :param y_tan: the y coordinate of escape route line and obstacle's tangent point
    :param i: the index of sheep
    :param k: the index of obs
    :return:
    """
    d_tan_bison = calculate_d(elude_bison[k][i][0], elude_bison[k][i][1], x_tan, y_tan)
    turn_v = math.sqrt(bison_v ** 2 - 10 * bison_v * (1 - math.cos(bison_angle[i]) ** 2) / math.cos(bison_angle[i]))
    print(turn_v)
    bison_x[i] = bison_x[i] + ((x_tan - elude_bison[k][i][0]) / d_tan_bison) * delta_t * turn_v
    bison_y[i] = bison_y[i] + ((y_tan - elude_bison[k][i][1]) / d_tan_bison) * delta_t * turn_v
    d[i] = calculate_d(bison_x[i], bison_y[i], wolf_x[0], wolf_y[0])  # 更新狼和羊群的距离
    update_d_bison_obs(i)
    if d_bison_obs[k][i] >= 2.5 * obstacles[k][2]:
        elude_flag[k][i] = True
        bison_flag[i] = False


def compare_angle(x1, y1, x2, y2, bison_i):
    theta1 = calculate_acos(x1, y1, bison_i)
    theta2 = calculate_acos(x2, y2, bison_i)
    list1 = [x1, y1, theta1]
    list2 = [x2, y2, theta2]
    if theta1 >= theta2:
        return list1
    else:
        return list2


def calculate_acos(x, y, bison_i):
    x_si = bison_x[bison_i]
    y_si = bison_y[bison_i]
    return math.acos(((x_si - wolf_x[0]) * (x - x_si) + (y_si - wolf_y[0]) * (y - y_si)) /
                     (math.sqrt((x_si - wolf_x[0]) ** 2 + (y_si - wolf_y[0]) ** 2) * math.sqrt(
                         (x - x_si) ** 2 + (y - y_si) ** 2)))


def update_d_wolf():
    for i in range(bison_n):
        d[i] = calculate_d(bison_x[i], bison_y[i], wolf_x[0], wolf_y[0])


def find_min_distance():
    """
    find the index of minimum distance in list of d
    :return:
    """
    return np.argmin(d)


def update_wolf():
    i = find_min_distance()
    if not bison_flag[i]:
        wolf_x[0] = wolf_x[0] + ((bison_x[i] - wolf_x[0]) / d[i]) * delta_t * wolf_v
        wolf_y[0] = wolf_y[0] + ((bison_y[i] - wolf_y[0]) / d[i]) * delta_t * wolf_v
        d[i] = calculate_d(bison_x[i], bison_y[i], wolf_x[0], wolf_y[0])
        update_d_wolf()
        if d[i] <= 0.6:
            kill(d[i])


def on_click(event):
    global wolf_flag
    update_bison()
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
    fig.canvas.mpl_connect('key_press_event', on_click)  # 设置键盘按键进行下一轮操作
    plt.clf()
    plt.scatter(bison_x, bison_y, s=10, c='lightblue', label='bison')  # 更新羊群坐标
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
    print(d_bison_obs)
    print(d)
    print(find_min_distance())
