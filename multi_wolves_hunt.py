import random
import math
import numpy as np
import matplotlib.pyplot as plt

bison_n: int = 10  # the amount of sheep
max_size = 3000  # map size
init_center = max_size / 2  # 初始化野牛群中心位置
init_distance = 500  # 初始化狼距离野牛群中心的位置
bison_x = []
bison_y = []
bison_flag = []  # 是否遇到障碍物
bison_elude_wolf_flag = []  # 野牛是否开始进入逃跑模式
all_bison_elude_wolf_flag = False  # 所有野牛进入逃跑模式标志
bison_v = 11.0  # the speed of sheep
bison_angle = []
elude_tan = []  # 进入躲避障碍物模式后设立的运动切点方向
elude_flag = []  # 进入躲避障碍物模式后是否逃脱的标志
elude_bison = []  # 进入躲避障碍物模式后保存野牛初始位置的坐标信息
predict_wolf = []
wolf_n = 3  # the amount of wolves
wolf_v = 12.6  # the speed of wolf
wolf_x = []  # 狼的横坐标
wolf_y = []  # 狼的纵坐标
wolf_stage2_round_count = 0  # 所有野牛速度提升至100后，狼速度提升至85%的回合数
round_count = 0  # 轮数计数器
delta_t = 0.5  # 时间间隔
d = []  # distance list between bison and wolf
wolf_flag: bool = False  # 狼的目标猎物是否进入躲避模式
wolf_group = [[], [], []]
wolf_group_flag = []
side_group_amount = 0
tail_group_amount = 0
surround_r = 5  # 包围半径

obs_n = 15
d_bison_obs = []  # 外层表示障碍物编号，内层表示与每一只野牛的距离
d_wolf_obs = []
obstacles = []  # the coordinates and radius list of obstacles

successful_flag = 0
successful_count = 0
max_round = 350
max_sim_times = 100
kill_rate = 0.1

fig, ax = plt.subplots()


def init():
    global d_bison_obs, elude_tan, elude_flag, elude_bison, d
    global side_group_amount, tail_group_amount, wolf_group_flag
    init_bison_x = init_center  # 初始化野牛中心点x
    init_bison_y = init_center  # 初始化野牛中心点y
    seed = np.random.randint(0, 2)
    if seed == 1:  # 分区域随机生成狼的位置
        init_wolf_x = np.random.randint(init_bison_x - init_distance, init_bison_x - 200)
    else:
        init_wolf_x = np.random.randint(init_bison_x + 200, init_bison_x + init_distance)
    seed = np.random.randint(0, 2)
    if seed == 1:
        init_wolf_y = np.random.randint(init_bison_y - init_distance, init_bison_y - 200)
    else:
        init_wolf_y = np.random.randint(init_bison_y + 200, init_bison_y + init_distance)
    side_group_amount = int(wolf_n/3)
    tail_group_amount = wolf_n - 2 * side_group_amount
    wolf_group_flag = [False, False, False]
    for i in range(wolf_n):
        x = np.random.randint(init_wolf_x - 10, init_wolf_x + 10)  # 在野牛群中心周围随机生成第i只狼x坐标
        wolf_x.append(x)
        y = np.random.randint(init_wolf_y - 10, init_wolf_y + 10)  # 在野牛群中心周围随机生成第i只狼x坐标
        wolf_y.append(y)
        predict_wolf.append((0, 0))  # 初始化狼处在预测模型时的初始坐标
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
        d_bison_obs.append([])
        elude_tan.append([])
        elude_flag.append([])
        elude_bison.append([])
        for j in range(bison_n):
            d_bison_obs[i].append(calculate_d(bison_x[j], bison_y[j], obstacles[i][0], obstacles[i][1]))
            elude_tan[i].append((0, 0))
            elude_flag[i].append(True)
            elude_bison[i].append((0, 0))
            # 初始化躲避模式相关参数
    # update_plt()


def find_min_distance(i):
    """
    find the wolf which is closest to bison i
    :return:
    """
    return np.argmin(d[i])


def find_min_distance_wolf():
    """
    find the bison which is closest in average to the wolves
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
        obs_flag = 0  # the flag of whether the bison encounter an obstacle
        p = find_min_distance(i)
        for j in range(obs_n):
            if d_bison_obs[j][i] <= 2.5 * obstacles[j][2]:
                bison_flag[i] = True  # 野牛i遇到了障碍物j
                if elude_flag[j][i]:
                    elude_obstacles(i, j, p)
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


def update_bison_normal(i):
    if not bison_elude_wolf_flag[i]:
        update_bison_stage1(i, 0.3)
    else:
        if not all_bison_elude_wolf_flag:
            update_bison_stage2(i)
        else:
            update_bison_stage3(i)


def update_bison_stage1(i, coe):  # basic function of update bison position
    j = find_min_distance(i)
    bison_x[i] = bison_x[i] + ((bison_x[i] - wolf_x[j]) / d[i][j]) * delta_t * bison_v * coe
    bison_y[i] = bison_y[i] + ((bison_y[i] - wolf_y[j]) / d[i][j]) * delta_t * bison_v * coe
    update_d_bison_wolf(i)
    update_d_bison_obs(i)


def update_bison_stage2(i):
    update_bison_stage1(i, 0.7)  # 当野牛与狼的最短距离为小于100时，将野牛的速度提高到70%


def update_bison_stage3(i):
    update_bison_stage1(i, 1)  # 当所有野牛进入逃离模式时，将野牛的速度提高到100%


def elude_obstacles(bison_i, k, p):
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
    a = (bison_y[bison_i] - wolf_y[p]) / (bison_x[bison_i] - wolf_x[p])
    b = -1
    c = bison_y[bison_i] - (a * bison_x[bison_i])
    d_obs_dir = math.fabs(a * xk + b * yk + c) / math.sqrt(a ** 2 + b ** 2)
    if d_obs_dir < r:  # 羊运动方向会撞上障碍物
        bison_flag[bison_i] = True
        a2 = -1 / (xk - bison_x[bison_i])
        b2 = yk - bison_y[bison_i]
        n1 = (-a2 ** 2 * b2 * r ** 2 + r * math.sqrt(math.fabs(a2 ** 2 * b2 ** 2 + 1 - a2 ** 2 * r ** 2))) / (
                a2 ** 2 * b2 ** 2 + 1)
        n2 = (-a2 ** 2 * b2 * r ** 2 - r * math.sqrt(math.fabs(a2 ** 2 * b2 ** 2 + 1 - a2 ** 2 * r ** 2))) / (
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
        target = find_min_distance_wolf()
        if bison_i == target:
            save_predict_wolf()
            update_wolf_predict(x, y, bison_i, k)
        elude_flag[k][bison_i] = False
    else:
        bison_flag[bison_i] = False
        elude_flag[k][bison_i] = True
        update_bison_normal(bison_i)


def surround_position_calculate(i, p):
    if 0 <= p < side_group_amount:
        x = bison_x[i] - surround_r
        y = bison_y[i]
        v = wolf_v + 1
        group_index = 0
    elif side_group_amount <= p < 2 * side_group_amount:
        x = bison_x[i] + surround_r
        y = bison_y[i]
        v = wolf_v + 1
        group_index = 1
    else:
        x = bison_x[i]
        y = bison_y[i] - surround_r
        v = wolf_v
        group_index = 2
    return x, y, v, group_index


def update_wolf_stage1(i, p, coe):  # basic function of update wolves position
    surround_x, surround_y, v, group_index = surround_position_calculate(i, p)
    d_surround = calculate_d(surround_x, surround_y, wolf_x[p], wolf_y[p])
    wolf_x[p] = wolf_x[p] + ((surround_x - wolf_x[p]) / d_surround) * delta_t * v * coe
    wolf_y[p] = wolf_y[p] + ((surround_y - wolf_y[p]) / d_surround) * delta_t * v * coe
    update_d_wolf_bison(p)
    if calculate_d(surround_x, surround_y, wolf_x[p], wolf_y[p]) < surround_r * (1.414 / 2):
        wolf_group_flag[group_index] = True
    else:
        wolf_group_flag[group_index] = False
    if (np.array(wolf_group_flag)).all():
        kill()


def update_wolf_stage2(i, p):
    update_wolf_stage1(i, p, 0.85)  # 所有野牛以70%的速度逃离后，将狼的速度提高到85%


def update_wolf_stage3(i, p):
    update_wolf_stage1(i, p, 1)  # 狼将速度提高到85%并运行一段时间后，将速度提高到100%


def predict_surround_position_calculate(i, p, k):
    if 0 <= p < side_group_amount:
        x = elude_bison[k][i][0] - surround_r / 2
        y = elude_bison[k][i][1]
        v = wolf_v + 1
        group_index = 0
    elif side_group_amount <= p < 2 * side_group_amount:
        x = elude_bison[k][i][0] + surround_r / 2
        y = elude_bison[k][i][1]
        v = wolf_v + 1
        group_index = 1
    else:
        x = elude_bison[k][i][0]
        y = elude_bison[k][i][1] - surround_r / 2
        v = wolf_v
        group_index = 2
    return x, y, v, group_index


def update_wolf_predict(x_tan, y_tan, i, k):
    """
    when sheep i encounters an obstacle, the wolf will make prediction
    """
    for p in range(wolf_n):
        surround_x, surround_y, v, group_index = predict_surround_position_calculate(i, p, k)
        d_temp = math.sqrt(math.fabs((2 * x_tan - surround_x - predict_wolf[p][0]) ** 2 + (2 * y_tan - surround_y
                                                                                 - predict_wolf[p][1]) ** 2))
        turn_v = math.sqrt(math.fabs(v ** 2 - 10 * v * (1 - math.cos(bison_angle[i]) ** 2) / math.cos(bison_angle[i])))
        wolf_x[p] = wolf_x[p] + (2 * x_tan - surround_x - predict_wolf[p][0]) / d_temp * delta_t * turn_v
        wolf_y[p] = wolf_y[p] + (2 * y_tan - surround_y - predict_wolf[p][1]) / d_temp * delta_t * turn_v
        update_d_wolf_bison(p)
        if calculate_d(surround_x, surround_y, wolf_x[p], wolf_y[p]) < surround_r * (1.414 / 2):
            wolf_group_flag[group_index] = True
        else:
            wolf_group_flag[group_index] = False
        if (np.array(wolf_group_flag)).all():
            kill()


def update_d_bison_wolf(i):
    for p in range(wolf_n):
        d[i][p] = calculate_d(bison_x[i], bison_y[i], wolf_x[p], wolf_y[p])  # 更新野牛和狼的距离


def update_d_wolf_bison(p):
    for i in range(bison_n):
        d[i][p] = calculate_d(bison_x[i], bison_y[i], wolf_x[p], wolf_y[p])  # 更新狼和野牛的距离


def update_d_bison_obs(bison_i):
    """
    update the list of d_bison_obs
    :param bison_i: the index of bison
    :return: NULL
    """
    for i in range(obs_n):
        d_bison_obs[i][bison_i] = calculate_d(bison_x[bison_i], bison_y[bison_i], obstacles[i][0], obstacles[i][1])


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
    turn_v = math.sqrt(
        math.fabs(bison_v ** 2 - 10 * bison_v * (1 - math.cos(bison_angle[i]) ** 2) / math.cos(bison_angle[i])))
    bison_x[i] = bison_x[i] + ((x_tan - elude_bison[k][i][0]) / d_tan_bison) * delta_t * turn_v
    bison_y[i] = bison_y[i] + ((y_tan - elude_bison[k][i][1]) / d_tan_bison) * delta_t * turn_v
    update_d_bison_wolf(i)  # 更新狼和野牛的距离
    update_d_bison_obs(i)
    if d_bison_obs[k][i] >= 2.5 * obstacles[k][2]:
        elude_flag[k][i] = True
        bison_flag[i] = False


def save_predict_wolf():
    for p in range(wolf_n):
        predict_wolf[p] = (wolf_x[p], wolf_y[p])


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


def kill():
    global successful_flag, successful_count
    rate = np.random.random()  # 随机生成一个捕杀成功率
    if rate < kill_rate:  # 根据资料显示狼捕杀成功率平均在14%
        # print("Successfully hunt! Round:", round_count)
        successful_flag = 1
        successful_count += 1


def on_click(event):
    # global wolf_flag
    # update_bison()
    # target = find_min_distance_wolf()
    # for i in range(obs_n):
    #     if not elude_flag[i][target]:
    #         update_wolf_predict(elude_tan[i][target][0], elude_tan[i][target][1], target, i)
    #         wolf_flag = True
    #         break
    # if not wolf_flag:
    #     update_wolf()
    # else:
    #     wolf_flag = False
    # update_plt()
    run()


def sim():
    global wolf_flag
    update_bison()
    target = find_min_distance_wolf()
    for i in range(obs_n):
        if not elude_flag[i][target]:
            update_wolf_predict(elude_tan[i][target][0], elude_tan[i][target][1], target, i)
            wolf_flag = True
            break
    if not wolf_flag:
        update_wolf()
    else:
        wolf_flag = False


def delete_list():
    global wolf_x, wolf_y, predict_wolf, wolf_n
    del bison_x[-bison_n:], bison_y[-bison_n:], bison_flag[-bison_n:], elude_bison[-bison_n:], bison_angle[-bison_n:]
    del wolf_x[-wolf_n:], wolf_y[-wolf_n:]
    del predict_wolf[-wolf_n:]
    del d[-bison_n:]
    del obstacles[-obs_n:], d_bison_obs[-obs_n:], elude_tan[-obs_n:], elude_flag[-obs_n:], elude_bison[-obs_n:]


def run():
    global successful_flag, successful_count, round_count, kill_rate, wolf_n
    for _ in range(10):
        print("wolf n", wolf_n)
        for _ in range(max_sim_times):
            init()
            round_count += 1
            for _ in range(max_round):
                if successful_flag == 0:
                    sim()
                else:
                    break
            delete_list()
            successful_flag = 0
        print(successful_count/max_sim_times*100, "%")
        successful_count = 0
        wolf_n += 1

def update_plt():
    global round_count
    round_count += 1
    fig.canvas.mpl_connect('key_press_event', on_click)  # 设置键盘按键进行下一轮操作
    plt.clf()
    plt.scatter(bison_x, bison_y, s=10, c='lightblue', label='bison')  # 更新野牛坐标
    plt.scatter(wolf_x, wolf_y, s=10, c='gray', label='wolf')  # 更新狼的坐标
    plt.scatter(obstacles[0][0], obstacles[0][1], s=obstacles[0][2] * 10, c='red', label='obstacles')
    for i in range(obs_n - 1):
        plt.scatter(obstacles[i + 1][0], obstacles[i + 1][1], s=obstacles[i + 1][2] ** 2 * np.pi * 0.1,
                    c='red')  # 更新野牛坐标
    plt.title("round " + str(round_count))
    plt.legend()
    plt.pause(120)


if __name__ == '__main__':
    run()
