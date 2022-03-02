import pstats
p = pstats.Stats('file.prof')
p.sort_stats('cumtime')
p.print_stats(5)