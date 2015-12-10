
## Copyright 2015 Tom Brown (FIAS, brown@fias.uni-frankfurt.de), Jonas Hoersch (FIAS, hoersch@fias.uni-frankfurt.de)

## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.


# make the code as Python 3 compatible as possible
from __future__ import print_function, division


__version__ = "0.1"
__author__ = "Tom Brown (FIAS), Jonas Hoersch (FIAS)"


#library for converting between projections
from pyproj import Proj, transform

import pandas as pd

import numpy as np

from operator import itemgetter


#%matplotlib inline

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap


bialek_file_name = "original-data/bialek.xlsx"


buses, lines, coords = itemgetter('Bus Records', 'Line Records', 'DisplayBus') \
                (pd.read_excel(bialek_file_name,
                       sheetname=None, skiprows=1, header=0))

buses.set_index('Number', inplace=True)
coords.set_index('Number', inplace=True)
lines.set_index(['From Number', 'To Number'], inplace=True)


coords.rename(columns={'X/Longitude Location': u'x_original',
                       'Y/Latitude Location': u'y_original'},
              inplace=True)


buses = pd.concat((buses, coords), axis=1)

#correct coordinates (lon,lat) of three buses
known_points = {1251: [-5.6027,36.0161], #Tarifa
                1257: [24.05,37.7], #Lavrio (Athens)
                1035: [8.5582,56.3508], #Idomlund
                }




#Main idea: assume the original points are linearly related to the
#projection of the ENTSO-E transmission network map (a Lambert
#Conformal Conical, see below)



#usual projection
usual = Proj(init='EPSG:4326')

#ENTSO-E map projection, see http://prj2epsg.org/epsg/3034 for more details
lcc = Proj(init='EPSG:3034')

lcc_points = {k: transform(usual,lcc,v[0],v[1]) for k,v in known_points.items()}

original_points = {k: buses.loc[k,["x_original","y_original"]].astype(float).values for k,v in known_points.items()}


def get_linear_transformation(points1,points2):
    """Given a dictionary of three points1 x that
    are known to map to three points2 y, find 2x2 matrix A
    and 2x1 vector b such that Ax+b = y for all pairs (x,y).
    Returns A and b."""

    #Must Solve B z = c where z are 6 parameters of A and b

    B = np.zeros((6,6))
    c = np.zeros((6))

    for i,(k,v) in enumerate(points1.items()):

        B[2*i,0] = v[0]
        B[2*i,1] = v[1]
        B[2*i,4] = 1.
        B[2*i+1,2] = v[0]
        B[2*i+1,3] = v[1]
        B[2*i+1,5] = 1.

        c[2*i] = points2[k][0]
        c[2*i+1] = points2[k][1]

    z = np.linalg.solve(B,c)

    A = z[:4].reshape((2,2))
    b = z[4:]

    return A,b



A,b = get_linear_transformation(original_points,lcc_points)


new_cols = ["x_lcc","y_lcc"]

old_cols = ["x_original","y_original"]


new = pd.DataFrame(data=np.dot(A, np.asarray(buses[old_cols]).T).T + b,
                   index = buses.index,
                   columns = new_cols)

buses = pd.concat((buses, new), axis=1)

buses["lon"], buses["lat"] = transform(lcc, usual, np.asarray(buses["x_lcc"]), np.asarray(buses["y_lcc"]))


#dump the results to file

buses_file_name = "final-data/buses.csv"

buses.to_csv(buses_file_name)

lines_file_name = "final-data/lines.csv"

lines.to_csv(lines_file_name)


#plot everything

fig,ax = plt.subplots(1,1)

fig.set_size_inches((7,7))

x1 = -10
x2 = 35
y1 = 35
y2 = 60


bmap = Basemap(resolution='i',projection='merc',llcrnrlat=y1,urcrnrlat=y2,llcrnrlon=x1,urcrnrlon=x2,ax=ax)

bmap.drawcountries()
bmap.drawcoastlines()


for (f,t),row in lines.iterrows():

    lon = [buses.loc[f,"lon"],buses.loc[t,"lon"]]
    lat = [buses.loc[f,"lat"],buses.loc[t,"lat"]]

    x,y = bmap(lon,lat)

    color = "g"
    alpha = 0.7
    width =  1.2

    bmap.plot(x,y,color,alpha=alpha,linewidth=width)

x,y = bmap(buses["lon"].values,buses["lat"].values)

highlight = known_points.keys()

bmap.scatter(x,y,color=["r" if u in highlight else "b" for u in buses.index],s=[50 if u in highlight  else 2 for u in buses.index])


fig.tight_layout()


file_name = "final-data/bialek-transformed.pdf"

print("file saved to",file_name)

fig.savefig(file_name)

plt.show()
