# initialize_counter.py

def initialize_counter():
    with open("counter.bin", "wb") as file:
        file.write((0).to_bytes(4, byteorder='big'))
    print("Counter file initialized.")

if __name__ == "__main__":
    initialize_counter()
