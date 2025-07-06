import numpy as np
from rust_annie import AnnIndex, Distance

def main():
    dim = 4

    # Initialize the index
    idx = AnnIndex(dim, Distance.COSINE)

    # Create some data
    data = np.random.rand(10, dim).astype(np.float32)
    ids  = np.arange(10, dtype=np.int64)

    # Add data to the index
    idx.add(data, ids)

    # Query the first vector
    q = data[0]
    neigh_ids, distances = idx.search(q, k=3)
    print("Query:", q)
    print("Neighbors:", neigh_ids)
    print("Distances:", distances)

    # Save & reload
    idx.save("example_index.bin")
    idx2 = AnnIndex.load("example_index.bin")
    neigh2, _ = idx2.search(q, k=3)
    print("Reloaded neighbors:", neigh2)

if __name__ == "__main__":
    main()
