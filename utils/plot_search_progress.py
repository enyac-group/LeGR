import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

filters = []
losses = []

with open(sys.argv[1], 'r') as f:
    for line in f:
        if 'Finish' in line:
            break
        if 'Filters left:' in line:
            filters.append(dict(eval(line[14:])))
        if 'Generation ' in line:
            losses.append(float(line[-6:]))

# initialization function: plot the background of each frame
def barlist(n):
    val = [filters[n][k] for k in filters[n]]
    return val

fig = plt.figure()
x = [k for k in filters[0]]
barcollection = plt.bar(x, barlist(0))
plt.xlabel('Layer index')
plt.ylabel('#Filters')
ylim = plt.gca().get_ylim()
loss_text = plt.text(0, int(ylim[1]*0.9), 'Minimum loss so far: {:.3f}'.format(losses[0]))

# animation function.  This is called sequentially
def animate(i):
    y = barlist(i)
    loss_text.set_text('Minimum loss so far: {:.3f}'.format(losses[i]))
    for i, b in enumerate(barcollection):
        b.set_height(y[i])

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, repeat=False, blit=False,
                               frames=len(filters), interval=500)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save(sys.argv[2], fps=2, extra_args=['-vcodec', 'libx264'])
