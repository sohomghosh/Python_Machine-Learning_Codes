from empty_package import a
from empty_package import b
from empty_package import data_reader

def main():
	print(a.a_func())
	#print(b)
	data=data_reader.get_data("empty_package/data/data_file1.csv")
	print(data.head())

if __name__ == '__main__':
	main()
