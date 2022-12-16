DPCXX = icpx
DOSXX = clang++

SOURCES += $(wildcard *.cpp)
DPCXX_TARGETS := $(patsubst %.cpp, %, $(SOURCES))
DOSXX_TARGETS := $(patsubst %.cpp, %-dosxx, $(SOURCES))
TARGETS := $(DPCXX_TARGETS) $(DOSXX_TARGETS)

CXXFLAGS = -std=c++20

DOSXXFLAGS = -std=c++20

LDLIBS =

DPCPP_FLAGS = -fsycl -lze_loader

all: $(TARGETS)

run: all
	@for target in $(foreach target,$(TARGETS),./$(target)) ; do echo "Running \"$$target\"" ; $$target ; done


dpcpp: $(DPCXX_TARGETS)

dosxx: $(DOSXX_TARGETS)

%: %.cpp
	$(DPCXX) $(CXXFLAGS) -o $@ $^ $(LD_FLAGS) $(DPCPP_FLAGS) $(LDLIBS)

%-dosxx: %.cpp
	$(DOSXX) $(CXXFLAGS) $(DOSXXFLAGS) -o $@ $^ $(LD_FLAGS) $(DPCPP_FLAGS) $(LDLIBS) -Wno-deprecated-declarations

clean:
	rm -fv $(TARGETS)
