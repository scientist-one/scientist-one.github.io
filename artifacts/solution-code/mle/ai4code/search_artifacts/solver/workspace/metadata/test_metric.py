from metric import kendall_tau

gt = [['a', 'b', 'c', 'd']]
pred1 = [['a', 'b', 'c', 'd']]
pred2 = [['a', 'c', 'b', 'd']]
pred3 = [['d', 'c', 'b', 'a']]

print("pred1:", kendall_tau(gt, pred1)) # 0 inversions -> 1
print("pred2:", kendall_tau(gt, pred2)) # 1 inversion -> 1 - 4 * 1 / 12 = 0.666
print("pred3:", kendall_tau(gt, pred3)) # 6 inversions -> 1 - 4 * 6 / 12 = -1
