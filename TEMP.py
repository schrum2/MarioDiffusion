import datetime as dt

# Temporary script to manually calculate the total time taken for two runs of a model training process for:
# Mar1and2-conditional-GTEsplit-negative0

# First run to checkpoint 320:
# {"epoch": 0, "loss": 6.236708337510073, "lr": 4.000000000000001e-06, "step": 216, "timestamp": "2025-06-09 07:48:19", "val_loss": 3.748273233572642}
# {"epoch": 320, "loss": 0.2043168224670269, "lr": 3.113347851168721e-05, "step": 69336, "timestamp": "2025-06-11 21:22:17", "val_loss": 0.21142561733722687}

"""
"2025-06-11 21:22:17" - "2025-06-09 07:48:19"
On 6/11: 21:22:17 hours
On 6/10: 24 hours
On 6/9: 16:11:41 hours
Total time: 61:34:38 hours
"""
# Resume to finish:
# {"epoch": 321, "loss": 0.204526297679102, "lr": 3.082764475205442e-05, "step": 69552, "timestamp": "2025-06-12 08:40:56"}
# {"epoch": 499, "loss": 0.20190688312329627, "lr": 0.0, "step": 108000, "timestamp": "2025-06-15 01:13:14", "val_loss": 0.2142385058104992}
"""
"2025-06-15 01:13:14" - "2025-06-12 08:40:56"
On 6/15: 1:13:14 hours
On 6/14: 24 hours
On 6/13: 24 hours
On 6/12: 15:32:18 hours
Total time: 64:45:32 hours
"""
# Total Time for both runs: 61:34:38 hours + 64:45:32 hours = 126:20:10 hours

first_run_start = "2025-06-09 07:48:19"
first_run_stop = "2025-06-11 21:22:17"
second_run_start = "2025-06-12 08:40:56"
second_run_stop = "2025-06-15 01:13:14"

#convert to seconds with datetime
first_run_start = dt.datetime.strptime(first_run_start, "%Y-%m-%d %H:%M:%S")
first_run_stop = dt.datetime.strptime(first_run_stop, "%Y-%m-%d %H:%M:%S")
second_run_start = dt.datetime.strptime(second_run_start, "%Y-%m-%d %H:%M:%S")
second_run_stop = dt.datetime.strptime(second_run_stop, "%Y-%m-%d %H:%M:%S")
                                          
# calculate total time in seconds
total_time = (first_run_stop - first_run_start) + (second_run_stop - second_run_start)
print(f"Total time in seconds: {total_time.total_seconds()}")
# calculate total time in hours
print(f"Total time in hours: {total_time.total_seconds() / 3600}")

# PRINTED OUTPUT:
# Total time in seconds: 453976.0
# Total time in hours: 126.10444444444444