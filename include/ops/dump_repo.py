import os

def dump_cuda_cpp_repo(repo_path, output_file):
    # source_extensions = ('.cc', '.cu', '.cuh', '.h', '.hpp', '.c', '.cpp')
    source_extensions = ('.cu', '.cuh')
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(source_extensions):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, repo_path)
                    
                    outfile.write(f"\n\n{'='*80}\n")
                    outfile.write(f"File: {relative_path}\n")
                    outfile.write(f"{'='*80}\n\n")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except UnicodeDecodeError:
                        outfile.write(f"Error: Unable to read {relative_path}. It might be a binary file.\n")

    print(f"Repository contents have been dumped to {output_file}")

# Usage example:
# dump_cuda_cpp_repo('/path/to/your/repo', 'output.txt')
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python dump_repo.py <repo_path> <output_file>")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    output_file = sys.argv[2]
    
    dump_cuda_cpp_repo(repo_path, output_file)
