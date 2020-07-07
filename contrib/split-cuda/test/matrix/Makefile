matrix_mul.o: matrix_mul.cu
	nvcc -g -Xcompiler  -g3 -O0 -c $< -o $@ 

matrix_mul.dummy.exe: matrix_mul.o
	/usr/bin/g++-8 $< -o $@ -L../../ -lcuda_wrappers 
	nvcc -g  $< -lcudart -o matrix_mul.exe  

compile: matrix_mul.o matrix_mul.dummy.exe

clean:
	rm -rf matrix_mul.dummy.exe matrix_mul.o matrix_mul.exe

