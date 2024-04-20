#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from math import *


# ## Longitude & Latitude of selected adressess

# Below we define a routine to extract the longitude and latitude coordinates of an address.

# In[10]:


from geopy.geocoders import Nominatim
from geopy.geocoders import Photon


def get_latitude_longitude(address):
    
    geolocator = Photon(user_agent="measurements")
    location = geolocator.geocode(address)
    
    if location:
        
        return (location.latitude, location.longitude)
    
    else:
        
        return (None, None)


# In[11]:


get_latitude_longitude("4395 Fulton Industrial Blvd, Atlanta, GA 30336")


# In[13]:


os.getcwd()


# **Select the cloud kitchen locations by gathering the addresses of 25 business buildings for lease in your chosen city; these locations must not be too close to one another (at least half a mile apart)**

# **Record the location indices and addresses in a file (in formats such as Excel or CSV), and subsequently load it from inside your Python program; avoid direct copy & pasting of the data. Then, obtain the longitude and latitude coordinates of the cloud kitchen locations using specialized Python libraries. If using the library “geopy” for this purpose, note that multiple accesses and extending the default timeout may be required to collect all coordinates.**

# # Cloud Kitchen Location Selection
# 
# In this notebook, we will select cloud kitchen locations by gathering the addresses of 25 business buildings for lease in Atlanta city. The chosen locations are not too close to one another (at least half a mile apart).
# 
# ## Steps:
# 1. Gather a list of 25 business building addresses for lease in Atlanta.
# 2. Geocode these addresses to obtain latitude and longitude coordinates.
# 3. Calculate distances between locations, we have also check all the distance are atleast 0.5 mile apart
# 
# 

# ## Data Loading and Preprocessing
# 
# In this section, we load a dataset containing addresses from a CSV file named "atlanta_geo_cordinate1.csv." We perform the following data preprocessing steps:
# 
# 1. **Load Addresses**: We use the `os` module to get the current working directory and construct the path to the CSV file. We then load the CSV file into a DataFrame `df`, where each row represents an address.
# 
# 2. **Column Renaming**: We rename the column in the DataFrame to "Address" for clarity.
# 
# 3. **Compute Latitude and Longitude**: We create a list of addresses from the DataFrame and use a function `get_latitude_longitude` to geocode each address, obtaining latitude and longitude coordinates. These coordinates are stored in a separate list called `lat_long`.
# 
# 4. **Create a Coordinates DataFrame**: We transform the `lat_long` list into a DataFrame named `coordinates`, with columns "longitude" and "latitude."
# 
# 5. **Concatenate DataFrames**: We concatenate the original address DataFrame `df` with the `coordinates` DataFrame to create a new DataFrame `data`. The resulting DataFrame has columns for street address, zip code, longitude, and latitude.
# 
# 6. **Data Cleanup**: We remove the word "Atlanta" from the "street address" column and strip trailing commas for consistency.
# 
# The code prepares the data for further analysis, such as spatial analysis or visualization, by obtaining the latitude and longitude coordinates for each address and cleaning up the address data.
# 
# 
# 

# In[58]:


# Load the addresses
path = os.path.join(os.getcwd(), "atlanta_geo_cordinate.csv")
df = pd.read_csv(path, header =None)
df.columns =["Address"]
df.head()


# In[61]:


# Compute the longitudes and latitudes
address_list = [each[0] for each in df.values.tolist()]
lat_long = [get_latitude_longitude(each_add) for each_add in address_list]

# Transform them in a dataframe
coordinates = pd.DataFrame(lat_long, columns = ["latitude", "longitude"])
coordinates = coordinates[["longitude", "latitude"]]
coordinates.head()


# In[62]:


# Check for NaN valus
coordinates.isna().sum()


# In[63]:


# Create the Zip columns
df = df["Address"].str.split("GA,", expand= True, n = 1)
df.head()


# In[64]:


# Concatenate with the addresses and output them in the dirctory
data = pd.concat((df, coordinates), axis = 1)
data.columns =  ["street address", "zip code", "longitude", "latitude"]

# Remove the State and trailing comas
data["street address"]  = data["street address"].str.replace("Atlanta", "").str.strip(", ")

data.head()


# In[65]:


# Output the dataframe to the directory
data.to_csv("atlanta_geo_data.csv", index = False)


# **Plot the locations of the cloud kitchens onto a 2-dimensional map, with the help of matplotlib. Draw the rectangular plot region so that it contains all 25 data-points within its boundaries, plus 2.5 miles east of the easternmost point, 2.5 miles west of the westernmost point, 2.5 miles north of the northernmost point, and 2.5 miles south of the southernmost point. Perform the requisite calculations in Python.**

# ## Visualizing Cloud Kitchen Locations on a Map
# 
# 
# ### Code Explanation:
# 
# 1. **Sample Cloud Kitchen Locations**: Replace the `cloud_kitchens` variable with our actual data, where each element is a tuple containing latitude and longitude coordinates representing a cloud kitchen location.
# 
# 2. **`adjust_boundaries` Function**: This function takes a list of locations and a specified radius in miles (`miles`). It calculates adjusted map boundaries based on the maximum and minimum latitude and longitude coordinates of the provided locations. The adjusted boundaries ensure that all cloud kitchen locations are visible on the map with a specified buffer (`2.5` miles in this case).
# 
# 3. **Calculating Adjusted Boundaries**: The `north`, `south`, `east`, and `west` variables store the adjusted boundaries by calling the `adjust_boundaries` function with the `cloud_kitchens` and `2.5` miles as arguments.
# 
# 4. **Plotting Cloud Kitchens**: The code uses Matplotlib and Seaborn to create a scatter plot of the cloud kitchen locations. It sets the x-axis as longitudes and the y-axis as latitudes, using a red color for the data points and labeling them as "Cloud Kitchens."
# 
# 5. **Adjusting Map Boundaries**: The map boundaries are adjusted using the `plt.xlim` and `plt.ylim` functions to ensure that all cloud kitchen locations are within the visible area of the map.
# 
# 6. **Labels and Title**: The code adds labels to the x-axis and y-axis, sets the title of the plot to "Cloud Kitchens Locations," adds a legend, and turns on the grid for better visualization.
# 
# 7. **Displaying the Map**: The `plt.show()` function is used to display the map with cloud kitchen locations.
# 
# This code segment allows us to visualize the distribution of cloud kitchen locations on a map and adjust the map boundaries to ensure that all locations are clearly visible within the specified buffer distance.
# 

# In[66]:


import matplotlib.pyplot as plt
from geopy.distance import geodesic

# Sample cloud kitchen locations (replace with your data)
# Format: [(latitude1, longitude1), (latitude2, longitude2), ...]
cloud_kitchens = lat_long

def adjust_boundaries(locations, miles):
    
    latitudes, longitudes = zip(*locations)

    northmost = max(latitudes)
    southmost = min(latitudes)
    eastmost = max(longitudes)
    westmost = min(longitudes)

    # Adjusting boundaries
    northmost_adjusted = geodesic(miles=miles).destination((northmost, eastmost), 0)[0]
    southmost_adjusted = geodesic(miles=-miles).destination((southmost, eastmost), 0)[0]
    eastmost_adjusted = geodesic(miles=miles).destination((northmost, eastmost), 90)[1]
    westmost_adjusted = geodesic(miles=-miles).destination((northmost, westmost), 90)[1]

    return northmost_adjusted, southmost_adjusted, eastmost_adjusted, westmost_adjusted

north, south, east, west = adjust_boundaries(cloud_kitchens, 2.5)

# Plotting the cloud kitchens
latitudes, longitudes = zip(*cloud_kitchens)

plt.figure(figsize=(10, 8))
sns.scatterplot(x = longitudes, y = latitudes, color='red', label='Cloud Kitchens')

plt.xlim(west, east)
plt.ylim(south, north)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Cloud Kitchens Locations')
plt.legend(frameon = False)
plt.grid(True)
plt.show()


# ## Load the data

# In[67]:


path = os.path.join(os.getcwd(), "atlanta_geo_data.csv")
data = pd.read_csv(path)
data.head()


# **Next, since the locations of the service stations are not yet known, you will generate a sample by randomly selecting 50 points from the cloud kitchen map (assume that service stations can be set up even if there is a lake or other geographical obstacle). Set a random seed to allow for replication of the sample.**

# ## Generating Random Station Points within Geographic Range
# 
# In this code snippet, random station points are generated within the geographic range defined by the minimum and maximum longitude and latitude values present in our dataset. The goal is to create a set of random station locations for potential use in geographical analyses or simulations.
# 
# ### Code Explanation:
# 
# - **Calculate Geographic Extents**: The code begins by calculating the minimum and maximum longitude and latitude values (`x_min`, `x_max`, `y_min`, `y_max`) present in our existing data. These values define the geographic boundaries within which the random station points will be generated.
# 
# - **Define Range Coordinates**: Two lists, `xy_min` and `xy_max`, are created to represent the minimum and maximum coordinates, respectively. These lists are formed based on the calculated minimum and maximum longitude and latitude values.
# 
# - **Set Number of Stations**: The variable `n` is defined to specify the number of random station points to generate. In this example, it is set to 50.
# 
# - **Generate Random Station Points**: Using NumPy's `np.random.uniform` function, random station points are generated within the geographic range specified by `xy_min` and `xy_max`. The `np.random.uniform` function creates random values uniformly distributed within a given range. The result is a NumPy array with shape `(n, 2)`, where each row represents a station's longitude and latitude coordinates.
# 
# - **Create a DataFrame**: Finally, a Pandas DataFrame named `station_points` is created to store the generated random station points. The columns of this DataFrame are labeled based on the columns in the original data that represent longitude and latitude.
# 
# The generated random station points can be used for various geographical applications, such as testing algorithms, simulating scenarios, or conducting spatial analyses within the specified geographic extent.
# 

# ## Generate the Station Points

# We sample the station points uniformly on the rectangle $\left[\text{longitude}_{\text{min}}, \text{longitude}_{\text{max}}\right] \text{x} \left[\text{latitude}_{\text{min}}, \text{latitude}_{\text{max}}\right]$
# 

# In[68]:


x_min = data["longitude"].min()
x_max = data["longitude"].max()

y_min = data["latitude"].min()
y_max = data["latitude"].max()


# List with the Min - Max coordinates
xy_min = [x_min, y_min]
xy_max = [x_max, y_max]

# Number of stations to sample
n = 50

station_points = np.random.uniform(xy_min, xy_max, (n, 2))

station_points = pd.DataFrame(station_points, columns = data.columns[2: ])
station_points.head()


# In[69]:


plt.figure(figsize=(10, 8))
sns.scatterplot(x = "longitude", y = "latitude", data = station_points
                , label = "Stations")
ax = sns.scatterplot(x = "longitude", y = "latitude", data = data, 
                s=70, label = "Cloud Kitchen")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Kitchens - Stations Map")
plt.legend(frameon = False)
# Save the figure
plt.savefig("Locations.jpeg")

plt.grid(True)
plt.show()


# **Construct a Python table of the collected location data using the “tabulate” library. The table should contain the cloud kitchen and service station indices along the first column (in other words, after the table header, the first 25 rows of the table will contain the cloud kitchen data, and the next 50 rows will contain the service station data). The other columns should be “Street Address”, “Zip Code”, and “Coordinates”; the address should be a string, the zip code an integer, and location a tuple containing the latitude and longitude (floats). The street address and zip code of service stations can be omitted.**
# 

# ## Data Merge and Transformation
# 
# In this section of code, we perform data merging and transformation operations on two data tables, namely "data" and "station_points." The goal is to create a merged dataset and perform some data cleaning and restructuring.
# 
# ### Code Explanation:
# 
# 1. **Outer Merge**: The code begins by performing an outer merge between the "data" and "station_points" tables using the `merge` function. An outer merge includes all rows from both tables, filling in missing values with NaN where necessary. The merged result is stored in the "out" DataFrame.
# 
# 2. **Truncating NaNs in "zip code" Column**: The "zip code" column in the merged DataFrame ("out") is checked for NaN values. Any NaN values are replaced with 0, and the column is then converted to integers. Finally, 0 values are replaced with "-" to represent missing or unknown zip codes.
# 
# 3. **Truncating NaNs in "street address" Column**: Similarly, the "street address" column is checked for NaN values. Any NaN values are replaced with 0 and then replaced with "-" to represent missing or unknown street addresses.
# 
# 4. **Adding an "Index" Column**: A new column named "Index" is added to the beginning of the "out" DataFrame. This column contains the original indices from both the "data" and "station_points" tables. It helps in tracking the source of each row in the merged dataset.
# 
# 5. **Creating a "Coordinates" Column**: Another new column named "Coordinates" is added after the "zip code" column. This column contains tuples containing longitude and latitude coordinates obtained from the "longitude" and "latitude" columns in the original tables. It allows for easy access to location coordinates for each row.
# 
# 6. **Dropping Original Longitude and Latitude Columns**: The original "longitude" and "latitude" columns are no longer needed in the merged DataFrame, so they are dropped using the `drop` method.
# 
# 7. **Column Renaming**: Finally, the column names of the merged DataFrame ("out") are updated to be more descriptive: "Index" for the original indices, "Street Address" for the street address, "Zip Code" for the zip code, and "Coordinates" for the location coordinates.
# 
# The result is a structured and cleaned DataFrame ("out") that combines information from both the "data" and "station_points" tables, making it easier to work with and analyze the data.
# 
# This code segment demonstrates data manipulation and restructuring techniques commonly used in data preprocessing tasks.
# 

# ## Locations.txt table

# In[70]:


# Outer merge on the kitchen and stations tables
out = data.merge(station_points, how = "outer")

# Truncate NaNs for the zip column
out["zip code"] = out["zip code"].fillna(0).astype(int).replace(0, "-")

# Truncate NaNs for the street address column
out["street address"] = out["street address"].fillna(0).replace(0, "-")

# Add a column with the original indices of the kitchens and the stations
out.insert(0, "Index", data.index.tolist() + station_points.index.tolist())

# Create the column with the tuples containing the coordinates
coordinates = [x for x in zip(out["longitude"], out["latitude"])]
out_columns = out.columns.tolist()
ind_to_add = out_columns.index("zip code")
out.insert(ind_to_add + 1, "Coordinates", coordinates)

# Drop the original long - lat columns
out.drop(["longitude", "latitude"], axis = 1, inplace = True)

# Rename
out.columns = ["Index", "Street Address", "Zip Code", "Coordinates"]
out


# In[71]:


from tabulate import tabulate

# Tabulate the dataframe
table = tabulate(out, headers='keys', tablefmt='pretty', showindex=False)
print(table)


# In[72]:


# Extract the table to a text file
with open('Locations.txt', 'w') as file:
    
    file.write(table)
    


# **Write a Python function called “distance” that calculates the Euclidean distance be- tween every cloud kitchen i ∈ I and every service location j ∈ J, given their lon- gitudes and latitudes. Use miles as the unit of measurement; use (and document in the code) appropriate sources used to determine how to translate distances between coordinates into miles. Apply this function to calculate and store the distances in a matrix [dij]i∈I,j∈J.**

# ## Haversine Distance Calculation Function
# 
# In this code, we have defined a Python function called `haversine_distance`. This function calculates the Haversine distance between two points, A and B, given their longitudes and latitudes in the form (longitude, latitude). The Haversine formula is commonly used to calculate the distance between two points on the Earth's surface using their geographic coordinates.
# 
# ### Code Explanation:
# 
# - **Function Description**: The function takes three arguments: `A`, `B`, and `r`. `A` and `B` are tuples representing the coordinates of the two points (longitude, latitude), and `r` is the radius of the Earth (or any other sphere) for which we want to calculate the distance.
# 
# - **Conversion to Radians**: The longitudes and latitudes are converted from degrees to radians using the `radians` function from the `math` module. This is necessary because the Haversine formula works with angles in radians.
# 
# - **Haversine Formula**: The Haversine formula computes the distance between two points on a sphere using their differences in latitude (`dlat`) and longitude (`dlong`). It involves several mathematical calculations, including the use of trigonometric functions like `sin`, `cos`, and `asin`.
# 
# - **Distance Calculation**: The final distance is calculated using the Haversine formula and is stored in the `distance` variable. This distance represents the shortest distance between points A and B on the surface of the Earth (or the sphere with radius `r`).
# 
# - **Return Statement**: The calculated distance is returned as the result of the function.
# 
# This code is useful for geospatial applications where we need to calculate distances between two geographic coordinates on the Earth's surface. The Haversine distance is commonly used in various location-based services and geographic information systems (GIS).
# 

# ### Distances given Longitudes and Latitudes

# Next we compute the Euclidean distances between each cloud kitchen and each station and store them in a $25 x 50$ array. 
# 
# To calculate the distance between two points given longitude $\theta$ and latitude $\phi$ coordinates, we write a function to compute the Haversine distance formula:
# $
# d = 2r \cdot \arcsin\left(\sqrt{\sin^2\left(\frac{\theta_2 - \theta_1}{2}\right) + \cos\theta_1 \cdot \cos\theta_2 \cdot \sin^2\left(\frac{\phi_2 - \phi_1}{2}\right)}\right)$
# 
# where:
# 
# $(\theta_1, \phi_1)$ and $(\theta_2, \phi_2)$ – Each point's coordinates;
# 
# $r$ – Earth's radius; and
# 
# $d$ – Great circle distance between the points.
# 

# In[73]:


def haversine_distance(A, B, r):
    
    """ Function to compute the  Haversine distance between 
    two points A, B given their longitudes and latitudes
    assuming that each points is represented in (longitude, latitude) form. """
    
    # Convert the longitutes and latitudes from degrees to radians
    long1, lat1, long2, lat2 = map(radians, [A[0], A[1], B[0], B[1]])
    
    dlat = lat2 - lat1
    dlong = long2 - long1
    
    inner_term = (sin(dlat / 2) ** 2) + cos(lat1) * cos(lat2) * (sin(dlong / 2) ** 2)
    
    distance = 2 * r * ((asin(sqrt(inner_term))))
                 
    return distance
    
    


# In[74]:


# Allocate empty array
distances = np.zeros((data.shape[0], station_points.shape[0]))

# Coordinates array of the kitchens
kitchen_coords = data.iloc[:, 2:]

# Earth's radius in miles
r = 3959

for i, A in kitchen_coords.iterrows():
    
    for j, B in station_points.iterrows():
        
        distances[i, j] = haversine_distance(A, B, r)
        
        


# In[75]:


# Extract the distances array:

np.savetxt("Distances.csv", distances, delimiter = ",")


# **Formulate the assignment problem specified below using the PuLP library. Make sure to use the same names for the parameters and variables.
# Parameters:
# Variables:
# i ∈ I Cloud kitchens j ∈ J Service stations
# dij Distance between cloud kitchen i and service station j
# 􏰀 1, if cloud kitchen i delivers to service station j zij : 0, otherwise.
# Objective function:
# min􏰁􏰁dijzij (Minimizetotaldistancetraveled) i∈I j∈J
# Constraints:
# 􏰁 zij = 1 for j ∈ J (Deliver to each service station) i∈I
# 􏰁 zij = 2 for i ∈ I (Deliver from each cloud kitchen to two service centers) j∈J
# 2
# • Solve and store the resulting assignments resulting from the solution using a sparse data structure.**

# ## Linear Programming Model for Facility Location Problem
# 
# In this code, a Linear Programming (LP) model is defined to solve the Facility Location Problem (FLP). The FLP is a classic optimization problem that involves deciding the locations for facilities to meet certain criteria while minimizing costs or maximizing benefits.
# 
# ### Code Explanation:
# 
# - **Model Initialization**: The code begins by initializing an LP model object named "model" using the `LpProblem` function from the PuLP library. The goal of this LP model is to maximize an objective function.
# 
# - **Decision Variables**: Decision variables are defined to represent the allocation of resources or facilities. In this case, `z` is a binary variable (0 or 1) that represents whether facility `i` serves location `j`. These variables are packed into an Lp dictionary.
# 
# - **Objective Function**: The objective function is defined using the `lpSum` function, which calculates the sum of the distances `d[i, j]` multiplied by the binary variables `z[i, j]` for all combinations of `i` and `j`. The objective is to maximize this sum, which typically represents a measure of efficiency, benefit, or profit.
# 
# - **Constraints**: Two sets of constraints are defined:
#   1. **Location Constraint**: For each facility `j` in set `J`, there must be exactly one location `i` in set `I` that serves it. This constraint ensures that each facility is assigned to one location.
#   2. **Allocation Constraint**: For each location `i` in set `I`, there must be exactly two facilities `j` in set `J` that serve it. This constraint ensures that each location is served by exactly two facilities.
# 
# - **Description**: It's important to note that the code does not provide specific details about the data used for this LP model. The code assumes the existence of data such as distances (`d`), sets of locations (`I`) and facilities (`J`), and their relationships.
# 
# Linear Programming models like this one are commonly used in operations research and logistics to solve facility location problems, transportation problems, and allocation problems. They help optimize resource allocation and decision-making in various real-world scenarios.
# 

# ## Linear Optimization with PuLP

# In[76]:


from pulp import *

# Initialize the model object
model = LpProblem("PPM-Generic", LpMaximize)

# Define the decision variables
I = list(range(data.shape[0]))
J = list(range(station_points.shape[0]))
d = distances

# Pack them in an Lp dictionary
z = LpVariable.dicts("z" ,((i, j) for i in I for j in J),
                     lowBound = 0, upBound = 1,
                     cat = LpBinary)

# Define the objective function
model += lpSum(d[i, j] * z[i, j] for i in I for j in J)

# Define the constraints
for j in J:
    
    model += lpSum(z[i, j] for i in I) == 1
    
for i in I:
    
    model += lpSum(z[i, j] for j in J) == 2
    
    


# In[77]:


# See the model spec
model


# In[78]:


# Solve the model
model.solve()


# In[79]:


# Extract the model spec in an MPS file
model.writeMPS("AP.mps")


# In[80]:


# Extract the decision matrix
solutions = np.zeros((len(I), len(J)))

for i in I:
    for j in J:
        
        solutions[i, j] = z[i, j].value()
        


# In[81]:


# Empty dictioanary to store the kitchen - stations pairs
od = {}

# Decision matrix
solutions = np.zeros((len(I), len(J)))

# Compute the total distance traveled
total_distance = 0

for i in I:
    
    for j in J:
        
        if z[i, j].value() == 1:
            
            # Extract the solutions matrix
            solutions[i, j] = z[i, j].value()
            
            # Create the kitchen - Station pair
            od[(i + 1, j + 1)] = distances[i, j]
            
            # Increment the distance traveled
            total_distance += distances[i, j]
                      
            
for (i, j), distance in od.items():
    
    print(f"Kitchen {i } delivers to station {j} with distance{distance: .2f} miles")


# In[45]:


print(f"Total distance traveled: {total_distance: .2f} miles")


# In[46]:


from scipy.sparse import csr_matrix, save_npz

# Convert the solutions matrix into csr sparse form
sparse_solutions = csr_matrix(solutions)

# Extract the sparse solutions matrix in .csr format
save_npz("Solution", sparse_solutions)


# In[47]:


sparse_solutions.data


# In[48]:


sparse_solutions.indices


# In[49]:


sparse_solutions.indptr


# **Task III
# • Build and populate an “Origin and Destination (OD)” table with the following columns “Cloud Kitchen Index (Origin),” “Service Station Index (Destination),” and “Distance (miles).”; make sure to print only the pairs actually selected by the model (i.e., with zij = 1).
# • Plot the assignment solution onto the map. You may use specialized Python libraries (“networkx”) or do this manually.
# • Create a frequency graph using for three different distance ranges (short, medium, long). The x-axis should have the distance ranges (e.g., < 3 miles, 3-6 miles, > 6 miles, and the y-axis should have the frequency values (e.g., % of Origin-Destination assignments within each range).**

# # Task III

# ## Building Origin-Destination (OD) Table for Cloud Kitchens and Service Stations
# 
# In this code, an Origin-Destination (OD) table is constructed to show the distances in miles between various pairs of cloud kitchens (origins) and service stations (destinations). The code uses the `PrettyTable` library to create a well-formatted table for easy visualization.
# 
# ### Code Explanation:
# 
# - **Table Initialization**: The code starts by initializing an empty table named "tbl" using the `PrettyTable` library. The table is intended to display the OD information.
# 
# - **Table Headers**: The headers for the table are set using the `tbl.field_names` attribute. The table will have three columns: "Cloud Kitchen Index (Origin)," "Service Station Index (Destination)," and "Distance (miles)."
# 
# - **Loop Over OD Dictionary**: The code then loops over the `od` dictionary, which likely contains pairs of indices (i, j) as keys and the corresponding distances in miles as values.
# 
# - **Adding Rows to the Table**: For each pair (i, j) in the OD dictionary, along with its associated distance, a new row is added to the table using the `tbl.add_row` method. This adds the information about the origin index, destination index, and distance to the table.
# 
# - **Table Printing**: Finally, the constructed OD table is printed to the console using the `print(tbl)` statement.
# 
# The resulting table provides a clear and organized view of the distances between cloud kitchens and service stations, making it easier to analyze and make decisions based on this information. It can be particularly useful in logistics and transportation planning to determine optimal routes or allocations.
# 

# ## OD Table

# In[51]:


# Building Origin & Destination OD Table
from prettytable import PrettyTable

tbl = PrettyTable()
tbl.field_names = ["Cloud Kitchen Index (Origin)", "Service Station Index (Destination)","Distance (miles)"]

for (i, j), distance in od.items():
    tbl.add_row([i, j, distance])
    
print(tbl)


# In[34]:


# Extract the OD table into a txt file
with open("OD.txt", "w") as file:
    
    file.write(str(tbl))
    


# ## Creating and Visualizing a Network Graph for Kitchen and Station Locations
# 
# In this code, a network graph is created and visualized to represent the locations of cloud kitchens and service stations. The code utilizes the NetworkX library for graph creation and Matplotlib for visualization.
# 
# ### Code Explanation:
# 
# - **Graph Initialization**: An empty undirected graph `G` is initialized using NetworkX. This graph will represent the network of kitchen and station locations.
# 
# - **Node Coordinates**: Coordinates for the kitchen and station nodes are extracted from the data and stored in `kitchen_nodes` and `station_nodes` lists, respectively. Each element in these lists is a tuple representing (longitude, latitude).
# 
# - **Combining Nodes**: All nodes, including both kitchen and station nodes, are combined into the `all_nodes` list.
# 
# - **Adding Nodes to Graph**: Nodes are added to the graph `G` with their respective labels. The loop iterates over all nodes in `all_nodes`, and each node is assigned a unique label.
# 
# - **Extracting Node Positions**: The positions (coordinates) of the nodes are extracted from the graph using `nx.get_node_attributes` and stored in the `pos` dictionary.
# 
# - **Node Colors**: Nodes are colored differently based on their type (kitchens in skyblue and stations in coral). The `kitchen_colors` and `station_colors` lists are created accordingly.
# 
# - **Extracting Edges**: The edges between kitchens and stations are extracted from the `od` dictionary and adjusted to match the node labels in the graph. These edges represent the connections between kitchens and stations.
# 
# - **Adding Edges to Graph**: Edges are added to the graph using `G.add_edges_from`. The edges connect the kitchen nodes to their corresponding station nodes.
# 
# - **Graph Visualization**: The graph is visualized using `nx.draw`. Node positions, labels, colors, and other styling options are specified. The resulting graph is displayed using Matplotlib.
# 
# - **Legends**: Legends are created to explain the node colors. Blue represents kitchens, and coral represents stations. Legends are added to the plot for clarity.
# 
# - **Saving the Plot**: The final plot is saved as "Solution.jpeg."
# 
# - **Displaying the Plot**: The plot is displayed using `plt.show()`.
# 
# This code generates a visual representation of the network connecting cloud kitchens with service stations. It provides insights into the relationships and connections between these locations, which can be useful for logistics and transportation planning.
# 

# ## Network Graph

# In[52]:


import networkx as nx

# Empty Graph
G = nx.Graph()

# List of tuples with the coords of the kitchens
kitchen_nodes = [(row[0], row[1]) for row in data.iloc[:, 2:].values]

# List of tuples with the coords of the stations
station_nodes = [(row[0], row[1]) for row in station_points.values]

# List of tuples with all the nodes
all_nodes = kitchen_nodes + station_nodes

# Empty Graph
G = nx.Graph()

# Iterate over all the points and add them to the graph vertices
# along with their respective labels
for i, node in enumerate(all_nodes):
    
    G.add_node(i + 1, pos = node)

# Extract the positions dictionary from the nodes
pos = nx.get_node_attributes(G, 'pos')

# Colors list to dicern between the kitchens and the stations
kitchen_colors = ["skyblue" for _ in kitchen_nodes]
station_colors = ["coral" for _ in station_nodes]
all_colors = kitchen_colors + station_colors

# Extract the kitchen - stations pairs to connect
edges_temp = list(od.keys())

# Increment the stations by 25 because we have them labeled from 1 - 75
edges = [ (x, y +25) for x, y in edges_temp]

# G.add_edges_from(edges)
G.add_edges_from(edges)

# Draw the resulting graph
fig, ax = plt.subplots(1, 1, figsize=(14, 12))

nx.draw(G, pos = pos, with_labels = True, 
        node_color= all_colors, font_size=10, ax = ax,
       node_size = 600)

ax.set_title("Solutions Map", fontsize = 14)

# Creating legends
legend_labels = {"skyblue": "Kitchens", "coral": "Stations"}
legend_elements = [plt.Line2D([0], [0], marker='o', 
                              color='w', markerfacecolor=color, 
                              markersize=10, label=label) for color, label in legend_labels.items()]

# Adding legends
ax.legend(handles=legend_elements, frameon = False, fontsize = 14)
plt.savefig("Solution.jpeg")

plt.show()


# # Frequency Graph

# In[54]:


# Extract the OD distances
od_distances = list(od.values())

# Bin Values
bins = [0, 3, 6 ,max(od_distances)]

# Frequency Graph
ax = sns.displot(od_distances, stat = "percent", bins = bins,
                height = 5, aspect = 1.4, color = "skyblue")

xticks = plt.xticks()[0]

yticks = plt.yticks()[0]
ylabs = [str(tick) + "%" for tick in yticks]
plt.yticks(ticks = yticks, labels = ylabs)

plt.xticks(ticks = [1.5, 4.5, (max(bins) + 6) / 2], 
           labels = ["<3 miles", "3-6 miles ", ">6 miles"], 
           fontsize = 12, rotation = 15)


plt.ylabel("% of Origin-Destination Assignments", fontsize = 12)
plt.title("Frequency Graph")
plt.savefig("Frequency.jpeg", transparent = True, bbox_inches='tight')
plt.show()


# In[ ]:




