CC=aarch64-linux-android21-clang
CFLAGS=-O3 -I../../../inference/engine/include -fopenmp
LDFLAGS=-fopenmp ../../../install_android-aarch64/lib/libbolt.a -llog \
	../../../third_party/android-aarch64/opencl/lib/libOpenCL.so \
	-lm -lstdc++

targets: c_image_classification c_input_method

c_image_classification: c_image_classification.o c_test.o
	${CC} -o $@ $^ ${LDFLAGS}
c_input_method: c_input_method.o c_test.o
	${CC} -o $@ $^ ${LDFLAGS}
%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@
clean:
	rm -rf *.o c_image_classification c_input_method
