import datetime as dt

first_run_start = "2025-06-09 07:48:19"
first_run_stop = "2025-06-11 21:34:00"
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
# Total time in seconds: 454679.0
# Total time in hours: 126.29972222222223