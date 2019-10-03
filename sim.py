import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import os
import matplotlib.cm as cm
cmap = cm.get_cmap('nipy_spectral')

import functions as func

from ipywidgets import Layout, Button, Box, FloatText, Textarea, Dropdown, Label, IntSlider
from ipywidgets import BoundedFloatText

import matplotlib.pyplot as plt
import numpy as np

from ipywidgets import widgets
from IPython.display import display,clear_output

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import matplotlib.cm as cm
cmap = cm.get_cmap('nipy_spectral')

def show_sim():
    plt.ioff()
    ax=plt.gca()
    out=widgets.Output()

    dy = 1
    tgrid_f = np.linspace(0,30,301)
    ygrid_f = np.arange(0,600,dy)


    layconc = Layout(width='40%')
    c0 = BoundedFloatText(value=1,min=0,max=1,step=1,layout=layconc)
    c1 = BoundedFloatText(value=0,min=0,max=1,step=1,layout=layconc)
    c2 = BoundedFloatText(value=0,min=0,max=1,step=1,layout=layconc)
    cell_items = Box([Label(value='cell'),c0,c1,c2],layout=Layout(flex_flow='column',width='8%'))

    a0 = BoundedFloatText(value=1,min=0,max=1,step=0.25,layout=layconc)
    a1 = BoundedFloatText(value=0,min=0,max=1,step=0.25,layout=layconc)
    a2 = BoundedFloatText(value=0,min=0,max=1,step=0.25,layout=layconc)
    chem_items = Box([Label(value='chem'),a0,a1,a2],layout=Layout(flex_flow='column',width='10%'))


    D_chem = BoundedFloatText(description='D chem', value=750,min=0,max=2000,step=50,layout=Layout(width='80%'))
    D_cell = BoundedFloatText(description='D cell', value=600,min=0,max=2000,step=50,layout=Layout(width='80%'))
    C_sens = BoundedFloatText(description=r'$\chi$', value=40,min=-100,max=100,step=5,layout=Layout(width='80%'))
    button=widgets.Button(description='Run sim')
    controls = widgets.Box([D_chem,D_cell,C_sens,button],layout=Layout(flex_flow='column',width='15%'))

    vbox=widgets.Box([cell_items,chem_items,out,controls])
    display(vbox)

    def click(b):
        ax.clear()
        vmax=0.025
        ca_f,cells_f = func.KS_rg( [D_chem.value,D_cell.value,C_sens.value*5e5], [a0.value,a1.value,a2.value, c0.value,c1.value,c2.value],tgrid_f,ygrid_f)
        ax.imshow(cells_f.clip(0,vmax*0.95).T,cmap=cmap,extent=(tgrid_f[0],tgrid_f[-1],ygrid_f[0],ygrid_f[-1]),vmin=0,vmax=vmax,aspect=1/50)
        ax.axhline(400,ls='--',color='red')
        ax.axhline(200,ls='--',color='red')
        fig = plt.gcf()
        fig.set_size_inches(16, 10.5)
        with out:
            clear_output(wait=True)
            display(ax.figure)

    button.on_click(click)
    click(None)
