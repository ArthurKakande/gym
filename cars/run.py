import warnings
warnings.filterwarnings("ignore")

from atime import *
from agent import *
from model import *
#from shapely.errors import TopologicalError


import pandas as pd
import numpy as np
#import geatpy as ea
import geopandas as gpd
#from sys import path as paths
#from os import path as path
import igraph as igp
import osmnx
import networkx as nx
#from scipy.spatial import cKDTree
#from mesa.space import NetworkGrid

'''load data'''
place = "Sioux Falls"
G = osmnx.graph_from_place(place, network_type="drive", simplify=False, retain_all=True, truncate_by_edge=True)
G = osmnx.project_graph(G)  # Ensure the graph is projected to a planar coordinate system
G = osmnx.add_edge_speeds(G)
G = osmnx.add_edge_travel_times(G)

nodes, edges = osmnx.graph_to_gdfs(G)

# Debug nodes and edges
print(nodes.columns)  # Should include 'geometry'
print(edges.columns)

# Recreate stations GeoDataFrame
stations = pd.read_csv("sixous_fall_alt_fuel_stations.csv")
point = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations.Longitude, stations.Latitude), crs="EPSG:4326")
#stations['osmid'] = point['geometry'].apply(lambda x: osmnx.get_nearest_node(G, (x.y, x.x)))
# Using vectorized nearest_nodes for efficiency
stations['osmid'] = osmnx.nearest_nodes(G, stations['Longitude'], stations['Latitude'])


#old code
#place = "Sioux Falls"
#G =  osmnx.graph_from_place(place,network_type='drive',simplify=False,retain_all = True,
#                                              truncate_by_edge = True) #clean_periphery=False
#osmnx.plot_graph(G)
#G = osmnx.add_edge_speeds(G)
#G = osmnx.add_edge_travel_times(G)

 
#Pass in a few default speed values (km/hour) to fill in edges with 
#missing `maxspeed` from OSM
#hwy_speeds = {"residential": 35, "secondary": 50, "tertiary": 60}
#G  = osmnx.add_edge_speeds(G , hwy_speeds)
#G = osmnx.add_edge_speeds(G)
#G  = osmnx.add_edge_travel_times(G )

nodes,edges = osmnx.graph_to_gdfs(G)

G_simplified = nx.DiGraph()
for u, v, data in G.edges(data=True):
    G_simplified.add_edge(u, v, length=data.get('length', 0), speed_kph=data.get('speed_kph', 0))
nx.write_gml(G_simplified, 'Sioux_Falls_simplified.gml')


#nx.write_gml(G, 'Sioux_Falls.gml')

'''read file'''
g = igp.read('Sioux_Falls_simplified.gml')
#G = nx.MultiDiGraph(nx.read_gml('Data\map.gml'))
#nodes,_ = osmnx.graph_to_gdfs(wa_G) 
#nodes_tree = cKDTree(np.transpose([nodes.geometry.x, nodes.geometry.y])) # 创建cKDTree减少计算量加速计算
#grid = NetworkGrid(G)

'''create stations geopandas file'''
#stations = pd.read_csv("sixous_fall_alt_fuel_stations.csv")
#point = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations.Longitude, stations.Latitude)) #指定经纬度坐标所在列
#point.crs = 'EPSG:4326' #Specify the coordinate system epsg4326 as WGS1984
#Find the osmid of the nearest node of stations and move the station to the road node
#stations['osmid'] = stations['geometry'].apply(lambda x: osmnx.get_nearest_node(G,(x.y,x.x)))
station_noodes = [nodes.index.get_loc(node) for node in stations.osmid]  #igraph id

 
#'''create ev geopandas file'''
#EV_agent = pd.read_csv("Sioux_Falls_veh_data.csv")
#d_point = gpd.GeoDataFrame(EV_agent, geometry = gpd.points_from_xy(EV_agent.dlng, EV_agent.dlat)) #指定经纬度坐标所在列
#EV_agent['dosmid'] = d_point['geometry'].apply(lambda x: osmnx.get_nearest_node(G,(x.y,x.x)))
#o_point = gpd.GeoDataFrame(EV_agent, geometry = gpd.points_from_xy(EV_agent.olng, EV_agent.olat)) #指定经纬度坐标所在列
#EV_agent['oosmid'] = o_point['geometry'].apply(lambda x: osmnx.get_nearest_node(G,(x.y,x.x)))

'''create ev geopandas file'''
# Load EV agent data
EV_agent = pd.read_csv("Sioux_Falls_veh_data.csv")

# Create GeoDataFrames for destination and origin points
d_point = gpd.GeoDataFrame(EV_agent, geometry=gpd.points_from_xy(EV_agent.dlng, EV_agent.dlat))
o_point = gpd.GeoDataFrame(EV_agent, geometry=gpd.points_from_xy(EV_agent.olng, EV_agent.olat))

# Update the CRS of the points to match the graph if necessary
d_point.crs = G.graph['crs']
o_point.crs = G.graph['crs']

# Find the nearest nodes for destinations
EV_agent['dosmid'] = d_point['geometry'].apply(
    lambda geom: osmnx.nearest_nodes(G, X=geom.x, Y=geom.y)
)

# Find the nearest nodes for origins
EV_agent['oosmid'] = o_point['geometry'].apply(
    lambda geom: osmnx.nearest_nodes(G, X=geom.x, Y=geom.y)
)


'''ABM model'''
EVmodel = EVModel('trail1', EV_agent = EV_agent, nodes_data = nodes,
                    stations = stations, iigraph = g, network = G, seed = 1)

df = EVmodel.run(steps = 4000)
df_EV = df[df.agent_type == 'EV']
df_EV_stranded = df_EV[df_EV.status == 'stranded']
df_EV_finished = df_EV[df_EV.status == 'finished']
df_EV_error = df_EV[df_EV.status == 'error'] #these vehicles can't find stations
df_EV_finished.distance_travelled  = df_EV_finished.distance_travelled/1000
#check an agent
#check_single_ev = df_EV.xs(681, level='AgentID') #multi index