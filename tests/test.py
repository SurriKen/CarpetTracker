x = [[600, 382, 732, 617], [652, 392, 752, 613], [813, 145, 943, 273]]
diagonal = (1920 ** 2 + 1080 ** 2) ** 0.5 * 0.1
print(diagonal)
dist = []
check = []
for i in x:
    for b in x:
        if b != i and sorted([b, i]) not in check:
            print(b, i)
            # distance = self.get_distance(self.track_list[i].boxes[-1], self.current_boxes[b])
            c1 = ((i[2] - i[0]) / 2, (i[3] - i[1]) / 2)
            c2 = ((b[2] - b[0]) / 2, (b[3] - b[1]) / 2)
            distance = ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
            # if distance < 0.05 * diagonal:
            dist.append((distance, i, b))
            check.append(sorted([b, i]))
dist = sorted(dist)
print('dist', dist, len(dist))