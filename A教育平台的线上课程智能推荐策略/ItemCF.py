import math
import random
from operator import itemgetter


class ItemBasedCF():
    def __init__(self, rcm=3, sim=10):
        # 找到相似的10门课程，为目标用户推荐3门课程
        self.n_sim_course = sim
        self.n_rec_course = rcm

        # 将数据集划分为训练集和测试集
        self.X_train = {}
        self.X_test = {}
        self.X = {}

        # 用户相似度矩阵
        self.course_sim_matrix = {}
        self.course_popular = {}
        self.course_count = 0

        print('Similar courses number = %d' % self.n_sim_course)
        print('Recommneded courses number = %d' % self.n_rec_course)

    # 读DF得到“用户-课程”数据
    def get_dataset_df(self, df, pivot=0.875):
        X_train_len = 0
        X_test_len = 0
        for i in df.index:
            user, course, rating = list(df.iloc[i])
            if(random.random() < pivot):
                # 相当于X_train.get(user)，若该键不存在，则设X_train[user] = {}，典中典
                self.X_train.setdefault(user, {})

                # 键中键：形如{'1': {'1287': '2.0', '1953': '4.0', '2105': '4.0'}, '2': {'10': '4.0', '62': '3.0'}}
                # 用户1看了id为1287的课程，打分2.0
                self.X_train[user][course] = rating
                X_train_len += 1
            else:
                self.X_test.setdefault(user, {})
                self.X_test[user][course] = rating
                X_test_len += 1
        print('X_train = %s' % X_train_len)
        print('X_test = %s' % X_test_len)

    # 读文件得到“用户-课程”数据
    def get_dataset(self, filename, pivot=0.875):
        X_train_len = 0
        X_test_len = 0
        for line in self.load_file(filename):
            user, course, rating = line.split(',')
            if(random.random() < pivot):
                # 相当于X_train.get(user)，若该键不存在，则设X_train[user] = {}，典中典
                self.X_train.setdefault(user, {})

                # 键中键：形如{'1': {'1287': '2.0', '1953': '4.0', '2105': '4.0'}, '2': {'10': '4.0', '62': '3.0'}}
                # 用户1看了id为1287的课程，打分2.0
                self.X_train[user][course] = rating
                X_train_len += 1
            else:
                self.X_test.setdefault(user, {})
                self.X_test[user][course] = rating
                X_test_len += 1
        print('X_train = %s' % X_train_len)
        print('X_test = %s' % X_test_len)

    # 读文件，返回文件的每一行
    def load_file(self, filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:  # 去掉文件第一行的title
                    continue
                yield line.strip('\r\n')

    # 计算课程之间的相似度
    def calc_course_sim(self):
        for user, courses in self.X_train.items():  # 循环取出一个用户和他看过的课程
            for course in courses:
                if course not in self.course_popular:
                    self.course_popular[course] = 0
                self.course_popular[course] += 1  # 统计每门课程共被看过的次数

        self.course_count = len(self.course_popular)  # 得到课程总数
        print("Total courses number = %d" % self.course_count)

        # 得到矩阵C，C[i][j]表示同时喜欢课程i和j的用户数
        for user, courses in self.X_train.items():
            for m1 in courses:
                for m2 in courses:
                    if m1 == m2:
                        continue
                    self.course_sim_matrix.setdefault(m1, {})
                    self.course_sim_matrix[m1].setdefault(m2, 0)
                    # self.course_sim_matrix[m1][m2] += 1  #同时喜欢课程m1和m2的用户+1    21.75  10.5   16.67
                    # ItemCF-IUF改进，惩罚了活跃用户 22.00 10.65 14.98
                    self.course_sim_matrix[m1][m2] += 1 / \
                        math.log(1 + len(courses))

        # 计算课程之间的相似性
        for m1, related_courses in self.course_sim_matrix.items():  # 课程m1，及m1这行对应的课程们
            for m2, count in related_courses.items():  # 课程m2 及 同时看了m1和m2的用户数
                # 注意0向量的处理，即某课程的用户数为0
                if self.course_popular[m1] == 0 or self.course_popular[m2] == 0:
                    self.course_sim_matrix[m1][m2] = 0
                else:
                    # 计算出课程m1和m2的相似度
                    self.course_sim_matrix[m1][m2] = count / math.sqrt(
                        self.course_popular[m1] * self.course_popular[m2])

        # 添加归一化
        maxDict = {}
        max = 0
        for m1, related_courses in self.course_sim_matrix.items():
            for m2, _ in related_courses.items():
                if self.course_sim_matrix[m1][m2] > max:
                    max = self.course_sim_matrix[m1][m2]

        for m1, related_courses in self.course_sim_matrix.items():  # 归一化
            for m2, _ in related_courses.items():
                # self.course_sim_matrix[m1][m2] = self.course_sim_matrix[m1][m2] / maxDict[m2]
                self.course_sim_matrix[m1][m2] = self.course_sim_matrix[m1][m2] / max

    # 针对目标用户U，找到K门相似的课程，并推荐其N门课程
    def recommend(self, user):
        K = self.n_sim_course  # 找到相似的20门课程
        N = self.n_rec_course  # 为用户推荐10门
        rank = {}
        watched_courses = self.X_train[user]  # 该用户看过的课程
#         X = self.X_train.copy().update(self.X_test)
#         watched_courses = self.X[user]  # 该用户看过的课程

        for course, rating in watched_courses.items():  # 遍历用户看过的课程及对其评价
            # 找到与course最相似的K门课程,遍历课程及与course相似度
            for related_course, w in sorted(self.course_sim_matrix[course].items(), key=itemgetter(1), reverse=True)[:K]:
                if related_course in watched_courses:  # 如果用户已经看过了，不推荐了
                    continue
                rank.setdefault(related_course, 0)
                rank[related_course] += w * float(rating)  # 计算用户对该课程的兴趣
        # 返回用户最感兴趣的N门课程
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    # 产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self):
        print('Evaluating...')
        N = self.n_rec_course  # 要推荐的课程数
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_courses = set()

        for i, user in enumerate(self.X_train):
            test_moives = self.X_test.get(user, {})  # 测试集中用户喜欢的课程
            rec_courses = self.recommend(user)  # 得到推荐的课程及计算出的用户对它们的兴趣

            for course, w in rec_courses:  # 遍历给user推荐的课程
                if course in test_moives:  # 测试集中有该课程
                    hit += 1  # 推荐命中+1
                all_rec_courses.add(course)
            rec_count += N
            test_count += len(test_moives)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_courses) / (1.0 * self.course_count)
        print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' %
              (precision, recall, coverage))


if __name__ == '__main__':
    rating_file = 'item_fc.csv'
    itemCF = ItemBasedCF()
    itemCF.get_dataset(rating_file)
    itemCF.calc_course_sim()
    itemCF.evaluate()
