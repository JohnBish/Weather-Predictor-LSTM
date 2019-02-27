from matplotlib import pyplot as plt

history = [
    {'loss': [8.221665335194263], 'acc': [0.14640291482350745]},
    {'loss': [8.141076699191968], 'acc': [0.2802401231905313]},
    {'loss': [6.646899270158313], 'acc': [0.29127140294848874]},
    {'loss': [6.648737461337314], 'acc': [0.29127140294848874]},
    {'loss': [7.839129571811244], 'acc': [0.29127140294848874]}
 ]

x, y1, y2 = range(len(history)), [i['loss'] for i in history], [i['acc'] for i in history]

fig = plt.figure()

a1 = fig.add_subplot(211)
a1.plot(x, y1)
a2 = fig.add_subplot(212)
a2.plot(x, y2) 
plt.show()