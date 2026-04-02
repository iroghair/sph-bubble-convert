import numpy as np
import sph_harm_functions as sph_harm
import os
import pyvista as pv
import pandas as pd
import itertools as it

def main(l_max,input,output):

   # Make new directory to save data
   os.makedirs(output,exist_ok=True)

   column_orb = ['orb_{}'.format(i) for i in range(0,(l_max+1)**2)]
   column_names = ['id', 'stl', 'sim', 'bub_num', 'time [s]', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'l_max'] + column_orb

   for folder in os.listdir(input):
      map = os.path.join(input, folder)
      sname = os.path.join(output, folder + '.csv')
      print('Currently working in: ', folder)

      # Create new dataframe
      df = pd.DataFrame(columns=column_names)

      for file in os.listdir(map):
         fname = os.path.join(map, file)
         print('Currently working on file ',file)

         if file.endswith(".pkl"):
            continue;

         df.loc[len(df)] = sph_harm_bubble(fname, l_max)

      # Get the velocities of the bubbles
      df = get_velocities(df)

      # Save dataframe
      df.to_csv(sname)

def sph_harm_bubble(fname, l_max):
   # Import stl file
   stl = pv.read(fname)

   # Get spherical harmonics from stl file
   weights, _ , _ = sph_harm.weights_from_stl(stl, rot=[0,0,0], l_max=l_max)   

   # Get relative path
   fdir = os.path.basename(os.path.dirname(fname))
   fname = os.path.basename(fname)
   fsave = os.path.join("data", "bubbles_stl", fdir, fname)
   
   # Get position of bubble
   pos = stl.center
   
   # Get the velocity of the bubble
   vel = np.array([0,0,0])
   
   # Get bubble number
   bub_num = int(fname.split('_')[-1].split('.')[0])

   # Set identifier
   id = fdir + '_' + str(bub_num)
   
   # Get time and convert to seconds
   time = float(fname.split('_')[0].lstrip('F'))
   time *= 1e-5 # s
   
   # Save results in dataframe
   data = [id, fsave, fdir, bub_num, time, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], l_max] + list(weights)

   return data
   
def get_velocities(df):
   for bubble in df['bub_num'].unique():
      # Get the velocities of the bubbles
      df_bubble = df[df['bub_num'] == bubble]
      df_bubble = df_bubble.sort_values(by=['time [s]'])
      df_bubble['vel_x'] = df_bubble['pos_x'].diff().shift(-1) / df_bubble['time [s]'].diff().shift(-1)
      df_bubble['vel_y'] = df_bubble['pos_y'].diff().shift(-1) / df_bubble['time [s]'].diff().shift(-1)
      df_bubble['vel_z'] = df_bubble['pos_z'].diff().shift(-1) / df_bubble['time [s]'].diff().shift(-1)
      df_bubble = df_bubble.fillna(0)
      df.update(df_bubble)
   
   return df

######################## START OF CODE ########################

# Set variables        
l_max = 14

# Set input and output folders
#loc = os.path.dirname(os.path.realpath(__file__))
#loc = os.path.join(loc, 'data')
loc = '/home/rachna/Simulations_new/6mm_eps10/start_24_end_27/'
input = os.path.join(loc,'bubbles_stl')
output = os.path.join(loc,'pickle_files_FT_new')

# file = "spherical-harmonics-dataset\\data\\pickle_files_FT\\4mm_eps05.csv"
# print('Currently working in: ', file)
# df = pd.read_csv(file)
# df2 = df.copy()
# get_velocities(df)


main(l_max, input, output)   
