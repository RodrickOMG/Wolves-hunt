import random
import math
import numpy as np
import matplotlib.pyplot as plt

bison_n: int = 10  # the amount of sheep
max_size = 2000  # map size
init_center = max_size / 2  # 初始化野牛群中心位置
init_distance = 100  # 初始化狼距离野牛群中心的位置
bison_x = []
bison_y = []
bison_flag = []  # 是否遇到障碍物
bison_elude_wolf_flag = []  # 野牛是否开始进入逃跑模式
all_bison_elude_wolf_flag = False  # 所有野牛进入逃跑模式标志
bison_v = 11.1  # the speed of sheep
bison_angle = []
elude_tan = []  # 进入躲避障碍物模式后设立的运动切点方向
elude_flag = []  # 进入躲避障碍物模式后是否逃脱的标志
elude_bison = []  # 进入躲避障碍物模式后保存野牛初始位置的坐标信息
predict_wolf = []
wolf_n = 5  # the amount of wolves
wolf_v = 13.6  # the speed of wolf
wolf_x = []  # 狼的横坐标
wolf_y = []  # 狼的纵坐标
wolf_stage2_round_count = 0  # 所有野牛速度提升至100后，狼速度提升至85%的回合数
round_count = 0  # 轮数计数器
delta_t = 0.5  # 时间间隔
d = []  # distance list between sheep and wolf
wolf_flag: bool = False  # 狼的目标猎物是否进入躲避模式

obs_n = 10
d_bison_obs = []  # 外层表示障碍物编号，内层表示与每一只野牛的距离
d_wolf_obs = []
obstacles = []  # the coordinates and radius list of obstacles

fig, ax = plt.subplots()


def init():
    global d_bison_obs, elude_tan, elude_flag, elude_bison, d
    init_bison_x = init_center  # 初始化野牛中心点x
    init_bison_y = init_center  # 初始化野牛中心点y
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
    for i in range(wolf_n):
        x = np.random.randint(init_wolf_x - 10, init_wolf_x + 10)  # 在野牛群中心周围随机生成第i只狼x坐标
        wolf_x.append(x)
        y = np.random.randint(init_wolf_y - 10, init_wolf_y + 10)  # 在野牛群中心周围随机生成第i只狼x坐标
        wolf_y.append(y)
    for i in range(bison_n):  # 随机生成羊群
        x = np.random.randint(init_bison_x - 5, init_bison_x + 5)  # 在野牛群中心周围随机生成第i只野牛x坐标
        bison_x.append(x)
        y = np.random.randint(init_bison_y - 5, init_bison_y + 5)  # 在野牛群中心周围随机生成第i只野牛y坐标
        bison_y.append(y)
        bison_flag.append(False)
        bison_elude_wolf_flag.append(False)
        bison_angle.append(0)
    for i in range(bison_n):
        d.append([])
        for j in range(wolf_n):
            d[i].append(calculate_d(bison_x[i], bison_y[i], wolf_x[j], wolf_y[j]))  # 初始化所有狼和羊的距离
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


def find_min_distance(i):
    """
    find the index of minimum distance in list of d
    :return:
    """
    return np.argmin(d[i])


def find_min_distance_wolf():
    """
    find the index of minimum distance in list of d
    :return:
    """
    temp_d = []
    temp_ave = []
    for i in range(bison_n):
        for p in range(wolf_n):
            temp_d.append(d[i][p])
        temp_ave.append(np.average(temp_d))
    return np.argmin(temp_ave)


def calculate_d(x1, y1, x2, y2):
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance


def check_bison_elude_wolf_flag():
    global all_bison_elude_wolf_flag
    flag = True
    for i in range(bison_n):
        if not bison_elude_wolf_flag[i]:
            flag = False
    all_bison_elude_wolf_flag = flag


def update_bison():
    global bison_elude_wolf_flag
    for i in range(bison_n):
        if not bison_elude_wolf_flag[i]:
            update_bison_stage1(i, 0.3)
        else:
            if not all_bison_elude_wolf_flag:
                update_bison_stage2(i)
            else:
                update_bison_stage3(i)
    check_bison_elude_wolf_flag()


def update_wolf():
    global wolf_stage2_round_count
    d_min_bison = find_min_distance_wolf()
    for p in range(wolf_n):
        if not all_bison_elude_wolf_flag:  # if some bison haven't been eluded state
            update_wolf_stage1(d_min_bison, p, 0.5)
        else:
            if wolf_stage2_round_count <= 5:
                update_wolf_stage2(d_min_bison, p)
                wolf_stage2_round_count += 1
            else:
                update_wolf_stage3(d_min_bison, p)
        for i in range(bison_n):  # if the distance between No.i bison and wolf less than 100, this bison should elude
            if d[i][p] <= 100:
                bison_elude_wolf_flag[i] = True


def update_bison_stage1(i, coe):  # basic function of update bison position
    j = find_min_distance(i)
    bison_x[i] = bison_x[i] + ((bison_x[i] - wolf_x[j]) / d[i][j]) * delta_t * bison_v * coe
    bison_y[i] = bison_y[i] + ((bison_y[i] - wolf_y[j]) / d[i][j]) * delta_t * bison_v * coe
    update_d_bison_wolf(i)


def update_bison_stage2(i):
    update_bison_stage1(i, 0.7)  # 当野牛与狼的最短距离为小于100时，将野牛的速度提高到70%


def update_bison_stage3(i):
    update_bison_stage1(i, 1)  # 当所有野牛进入逃离模式时，将野牛的速度提高到100%


def update_wolf_stage1(i, p, coe):  # basic function of update wolves position
    wolf_x[p] = wolf_x[p] + ((bison_x[i] - wolf_x[p]) / d[i][p]) * delta_t * wolf_v * coe
    wolf_y[p] = wolf_y[p] + ((bison_y[i] - wolf_y[p]) / d[i][p]) * delta_t * wolf_v * coe
    update_d_wolf_bison(p)


def update_wolf_stage2(i, p):
    update_wolf_stage1(i, p, 0.85)  # 所有野牛以70%的速度逃离后，将狼的速度提高到85%


def update_wolf_stage3(i, p):
    update_wolf_stage1(i, p, 1)  # 狼将速度提高到85%并运行一段时间后，将速度提高到100%


def update_d_bison_wolf(i):
    for p in range(wolf_n):
        d[i][p] = calculate_d(bison_x[i], bison_y[i], wolf_x[p], wolf_y[p])  # 更新野牛和狼的距离


def update_d_wolf_bison(p):
    for i in range(bison_n):
        d[i][p] = calculate_d(bison_x[i], bison_y[i], wolf_x[p], wolf_y[p])  # 更新狼和野牛的距离


def on_click(event):
    update_bison()
    update_wolf()
    update_plt()


def update_plt():
    global round_count
    round_count += 1
    fig.canvas.mpl_connect('key_press_event', on_click)  # 设置键盘按键进行下一轮操作
    plt.clf()
    plt.scatter(bison_x, bison_y, s=10, c='lightblue', label='bison')  # 更新羊群坐标
    plt.scatter(wolf_x, wolf_y, s=10, c='gray', label='wolf')  # 更新狼的坐标
    # for p in range(wolf_n):
    #     plt.annotate(p, xy=(wolf_x[p], wolf_y[p]), xytext=(wolf_x[p]+0.1, wolf_y[p]+0.1))
    plt.scatter(obstacles[0][0], obstacles[0][1], s=obstacles[0][2] * 10, c='red', label='obstacles')
    for i in range(obs_n - 1):
        plt.scatter(obstacles[i + 1][0], obstacles[i + 1][1], s=obstacles[i + 1][2] ** 2 * np.pi * 0.1,
                    c='red')  # 更新野牛坐标
    plt.title("round " + str(round_count))
    plt.legend()
    plt.pause(120)


if __name__ == '__main__':
    init()
    print(d)
