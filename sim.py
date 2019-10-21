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
from ipywidgets import GridspecLayout, HBox

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
    button=widgets.Button(description='Run sim',layout=Layout(width='80%'))
    controls = widgets.Box([D_chem,D_cell,C_sens,button],layout=Layout(flex_flow='column',width='20%'))

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
        fig.set_size_inches(10, 10.5)
        with out:
            clear_output(wait=True)
            display(ax.figure)

    button.on_click(click)
    click(None)


def create_expanded_button(description):
    return Button(description=description, layout=Layout(height='auto', width='auto'))
def create_expanded_float(description, button_style,min=0,max=1):
    return BoundedFloatText(value=0,min=0,max=1, step=0.05, layout=Layout(height='80%', width='80%'))

def show_sim2():
    import functions as func	 
    plt.ioff()
    ax=plt.gca()
    out=widgets.Output()


    dy = 1
    tgrid_f = np.linspace(0,30,301)
    ygrid_f = np.arange(0,600,dy)


    grid = GridspecLayout(20, 10, height='100%')

    for i in range(3):
        for j in range(3):
            grid[1+i,1+j]=create_expanded_float('test',None)
    grid[2,1].value=1
    grid[2,2].value=1
    grid[1:4,0]=create_expanded_button("Conc")


    grid[4,0]=create_expanded_button("Diffusion const")

    D_cell =create_expanded_float('Diffusion const cells',None)
    for j in range(3):
        grid[4,1+j]=BoundedFloatText(min=0,max=25000,step=100,value=2000,layout=Layout(width='80%'))
    grid[4,1].value=2000
    grid[4,2].value= 500
    grid[4,3].value=2400

    grid[5,0]=create_expanded_button("Sensitivity")
    for j in range(2):
        grid[5,2+j]=BoundedFloatText(min=-50,max=50,step=1,value=10,layout=Layout(width='80%'))
    button=widgets.Button(description='Run sim')
    grid[5,6]=button


    grid[0,1]=HBox(value="Cells")
    grid[0,2]=Label(value="Chemical 1")
    grid[0,3]=Label(value="Chemical 2")


    grid[6:,:]=out

    display(grid)


    def click(b):

        D_cell = grid[4,1].value
        D_1    = grid[4,2].value
        D_2    = grid[4,3].value

        S_1    = grid[5,2].value
        S_2    = grid[5,3].value

        c_cell= [ grid[i,1].value for i in range(1,4) ]
        c_1   = [ grid[i,2].value for i in range(1,4) ]
        c_2   = [ grid[i,3].value for i in range(1,4) ]


        cells_f,ca_f = func.KS_rg_gen( [D_cell, D_1, S_1*5E6, D_2, S_2*5E6], [*c_cell,*c_1, *c_2],tgrid_f,ygrid_f)

        ax.clear()
        vmax=0.025
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
