# import joblib
# import time, math
#
#
#
# def my_fun(i):
#     """ We define a simple function here.
#        """
#     time.sleep(1)
#     return math.sqrt(i ** 2)
#
# num = 10
# # normal
# start = time.time()
# for i in range(num):
#     my_fun(i)
#
# end = time.time()
#
# print('{:.4f} s'.format(end - start))
#
# #joblib
# start = time.time()
# # n_jobs is the number of parallel jobs
# # n_job = 并行工作个数
# joblib.Parallel(n_jobs=2)(joblib.delayed(my_fun)(i) for i in range(num))
# end = time.time()
# print('{:.4f} s'.format(end-start))