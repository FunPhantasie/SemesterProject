from mhd2dLib import MHD2D
import time as clock

N = 512
nu = 1.e-3 # normal viscosity
# nu = 1e-8 # hyperviscosity viscosity
# nu = 1 # normal viscosity

t_end = 1.5
t_print = 0.01

t = 0.
t_out = 0.
out_stats = 0
num_stats = 10

out_dir = "test_OT"

sim = MHD2D(N, nu, out_dir)

sim.print_vtk()
start_time = clock.time()
while(t < t_end):
  
  dt = sim.stepEuler()
  t += dt
  t_out += dt  
  
  # if (out_stats > num_stats):
  #   sim.print_stats()
  #   out_stats -= num_stats
  
  if(t_out>t_print):
    
    sim.print_vtk()
    # sim.print_spectrum()
    t_out -= t_print
  
  out_stats += 1
    
duration = clock.time() - start_time
print('Duration =', duration, '[s]')
