import pstats

# Load the binary profile data
p = pstats.Stats('profile.txt')

# Sort by 'tottime' or 'cumulative' and print the top 10 rows
print("--- Top 10 functions by Total Time ---")
p.sort_stats('tottime').print_stats(10)

print("\n--- Top 10 functions by Cumulative Time ---")
p.sort_stats('cumulative').print_stats(10)