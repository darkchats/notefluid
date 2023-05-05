import matplotlib.pyplot as plt
import pandas as pd

from notefluid.ellipse.utils.largs_plot import largs_plot_speed

data3 = """
u_theta -0.0100922 0 19.4988 609.95 0.00669423 0.00134881
u_theta -0.0276957 1 18.5653 609.579 0.0176874 0.0105962
u_theta -0.0398233 2 17.7752 608.956 0.0237759 0.023626
u_theta -0.0464439 3 17.1266 608.184 0.0253393 0.0355866
u_theta -0.0484408 4 16.5965 607.326 0.0236576 0.0439643
u_theta -0.0467298 5 16.1641 606.414 0.0199827 0.0478002
u_theta -0.042086 6 15.8137 605.468 0.0153422 0.0469822
u_theta -0.0351526 7 15.5345 604.499 0.0105428 0.0418594
u_theta -0.0264692 8 15.3187 603.513 0.00619908 0.0330429
u_theta -0.0165106 9 15.1609 602.516 0.00276982 0.0213057
u_theta -0.00569935 10 15.0575 601.513 0.000574686 0.0075117
u_theta 0.00556571 11 15.0064 600.505 -0.000187245 -0.00741149
u_theta 0.0169056 12 15.0064 599.495 0.00056875 -0.0225121
u_theta 0.0279405 13 15.0575 598.487 0.00281734 -0.0368254
u_theta 0.0382865 14 15.1609 597.484 0.00642296 -0.049406
u_theta 0.0475352 15 15.3187 596.487 0.0111327 -0.0593406
u_theta 0.0552472 16 15.5345 595.501 0.0165694 -0.0657879
u_theta 0.0609242 17 15.8137 594.532 0.0222095 -0.068012
u_theta 0.0639913 18 16.1641 593.586 0.0273641 -0.0654571
u_theta 0.0637564 19 16.5965 592.674 0.0311374 -0.0578644
u_theta 0.059374 20 17.1266 591.816 0.0323938 -0.045494
u_theta 0.0498347 21 17.7752 591.044 0.0297531 -0.0295656
u_theta 0.034152 22 18.5653 590.421 0.0218105 -0.0130664
u_theta 0.0123475 23 19.4988 590.05 0.00819021 -0.00165023
u_theta -0.0123475 24 20.5012 590.05 -0.00819021 -0.00165023
u_theta -0.034152 25 21.4347 590.421 -0.0218105 -0.0130664
u_theta -0.0498347 26 22.2248 591.044 -0.0297531 -0.0295656
u_theta -0.059374 27 22.8734 591.816 -0.0323938 -0.045494
u_theta -0.0637564 28 23.4035 592.674 -0.0311374 -0.0578644
u_theta -0.0639913 29 23.8359 593.586 -0.0273641 -0.0654571
u_theta -0.0609242 30 24.1863 594.532 -0.0222095 -0.068012
u_theta -0.0552472 31 24.4655 595.501 -0.0165694 -0.0657879
u_theta -0.0475352 32 24.6813 596.487 -0.0111327 -0.0593406
u_theta -0.0382865 33 24.8391 597.484 -0.00642296 -0.049406
u_theta -0.0279405 34 24.9425 598.487 -0.00281734 -0.0368254
u_theta -0.0169056 35 24.9936 599.495 -0.00056875 -0.0225121
u_theta -0.00556571 36 24.9936 600.505 0.000187245 -0.00741149
u_theta 0.00569935 37 24.9425 601.513 -0.000574686 0.0075117
u_theta 0.0165106 38 24.8391 602.516 -0.00276982 0.0213057
u_theta 0.0264692 39 24.6813 603.513 -0.00619908 0.0330429
u_theta 0.0351526 40 24.4655 604.499 -0.0105428 0.0418594
u_theta 0.042086 41 24.1863 605.468 -0.0153422 0.0469822
u_theta 0.0467298 42 23.8359 606.414 -0.0199827 0.0478002
u_theta 0.0484408 43 23.4035 607.326 -0.0236576 0.0439643
u_theta 0.0464439 44 22.8734 608.184 -0.0253393 0.0355866
u_theta 0.0398233 45 22.2248 608.956 -0.0237759 0.023626
u_theta 0.0276957 46 21.4347 609.579 -0.0176874 0.0105962
u_theta 0.0100922 47 20.5012 609.95 -0.00669423 0.00134881
"""

data = data3
d2 = [[float(i) for i in line.split(" ")[1:]] for line in data.split("\n") if len(line) > 10]
df = pd.DataFrame(d2)
df.columns = ['us', 'l', 'lx', 'ly', 'ux', 'uy']
largs_plot_speed(df)
largs_plot_speed(df)
plt.show()
