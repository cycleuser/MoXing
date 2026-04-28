import json

with open('llama_cpp_comprehensive_results.json', 'r') as f:
    results = json.load(f)

# Count successes
success_count = sum(1 for r in results if r.get('success'))
fail_count = sum(1 for r in results if not r.get('success'))

print(f'Total: {len(results)}')
print(f'Success: {success_count}')
print(f'Fail: {fail_count}')

# Show successful tests
print('\nSuccessful tests:')
for r in results:
    if r.get('success'):
        print(f"  {r['model']} - {r['config']}: {r['tokens_per_sec']:.2f} tok/s")

# Show failed tests
print('\nFailed tests:')
for r in results:
    if not r.get('success'):
        print(f"  {r['model']} - {r['config']}: {r.get('error', 'Unknown')}")
