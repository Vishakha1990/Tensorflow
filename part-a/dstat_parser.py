import sys

with open(sys.argv[1]) as f:
	avg_idle_cpu = 0.0
	max_idle_cpu = 0.0
	min_idle_cpu = 100.0
	
	mem_used = 0.0
	mem_free = 0.0
	
	disk_read = 0.0
	disk_write = 0.0
	
	nw_rec  = 0.0
	nw_send = 0.0
	
	count = 0
	
	for line in f:
		
		if not line.startswith('-') and not (line.strip().startswith('time')):
			
			count = count + 1
			
			temp = line.split('|')
			
			# CPU
			cpu = temp[1].split()
			avg_idle_cpu = avg_idle_cpu + float(cpu[2])
			if float(cpu[2]) > max_idle_cpu:
				max_idle_cpu = float(cpu[2]) 
			
			if float(cpu[2]) < min_idle_cpu:
				min_idle_cpu = float(cpu[2])
				
			# MEMORY
			memory = temp[2].split()
			if memory[0].endswith('M'):
				mem_used = mem_used + float(memory[0][:-1])
			elif memory[0].endswith('B'):
				mem_used = mem_used + float(memory[0][:-1])/(1024*1024)
			elif memory[0].endswith('G'):
				mem_used = mem_used + float(memory[0][:-1])*1024
			elif memory[0].endswith('k'):
				mem_used = mem_used + float(memory[0][:-1])/(1024)
				
				
			if memory[3].endswith('M'):
				mem_free = mem_free + float(memory[3][:-1])
			elif memory[3].endswith('B'):
				mem_free = mem_free + float(memory[3][:-1])/(1024*1024)
			elif memory[3].endswith('G'):
				mem_free = mem_free + float(memory[3][:-1])*1024	
			elif memory[3].endswith('k'):
				mem_free = mem_free + float(memory[3][:-1])/1024
			
			# DISK
			disk = temp[4].split()
			if disk[0].endswith('M'):
				disk_read = disk_read + float(disk[0][:-1])
			elif disk[0].endswith('B'):
				disk_read = disk_read + float(disk[0][:-1])/(1024*1024)
			elif disk[0].endswith('G'):
				disk_read = disk_read + float(disk[0][:-1])*1024
			elif disk[0].endswith('k'):
				disk_read = disk_read + float(disk[0][:-1])/(1024)

			if disk[1].endswith('M'):
				disk_write = disk_write + float(disk[1][:-1])
			elif disk[1].endswith('B'):
				disk_write = disk_write + float(disk[1][:-1])/(1024*1024)
			elif disk[1].endswith('G'):
				disk_write = disk_write + float(disk[1][:-1])*1024
			elif disk[1].endswith('k'):
				disk_write = disk_write + float(disk[1][:-1])/(1024)
			
			
			# NW
			nw = temp[5].split()
			if nw[0].endswith('M'):
				nw_rec = nw_rec + float(nw[0][:-1])
			elif nw[0].endswith('B'):
				nw_rec = nw_rec + float(nw[0][:-1])/(1024*1024)
			elif nw[0].endswith('G'):
				nw_rec = nw_rec + float(nw[0][:-1])*1024
			elif nw[0].endswith('k'):
				nw_rec = nw_rec + float(nw[0][:-1])/1024
				
				
			if nw[1].endswith('M'):
				nw_send = nw_send + float(nw[1][:-1])
			elif nw[1].endswith('B'):
				nw_send = nw_send + float(nw[1][:-1])/(1024*1024)
			elif nw[1].endswith('G'):
				nw_send = nw_send + float(nw[1][:-1])*(1024)
			elif nw[1].endswith('k'):
				nw_send = nw_send + float(nw[1][:-1])/(1024)
			
			#print float(cpu[2])
			#print float(memory[0][:-1])
			#print float(memory[3][:-1])
			#print nw[0][:-1]
			#print nw[1][:-1]
			
	print 'avg_idle_cpu: ' + str(avg_idle_cpu/count)
	print 'max_idle_cpu: ' + str(max_idle_cpu)	
	print 'min_idle_cpu: ' + str(min_idle_cpu)	
	
	print 'avg_mem_used: ' + str(mem_used/(count)) + ' MB'
	print 'avg_mem_free: ' + str(mem_free/count) + ' MB'
	
	print 'total disk_read: ' + str(disk_read) + ' MB'
	print 'total disk_write: ' + str(disk_write) + ' MB'
	
	print 'total nw_rec: ' + str(nw_rec) + ' MB'
	print 'total nw_send: ' + str(nw_send) + ' MB'
				