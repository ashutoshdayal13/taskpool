# make test | make tsan | make asan | make bench
CXX      ?= g++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -Iinclude -pthread

test:
	$(CXX) $(CXXFLAGS) tests/tests.cpp -o tests_bin && ./tests_bin

tsan:
	$(CXX) $(CXXFLAGS) -O1 -g -fsanitize=thread tests/tests.cpp -o tests_tsan && \
	./tests_tsan

asan:
	$(CXX) $(CXXFLAGS) -O1 -g -fsanitize=address,undefined tests/tests.cpp -o tests_asan && ./tests_asan

bench:
	$(CXX) $(CXXFLAGS) bench/bench.cpp -o bench_bin && ./bench_bin

clean:
	rm -f tests_bin tests_tsan tests_asan bench_bin

.PHONY: test tsan asan bench clean
