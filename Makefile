CXX      = g++
CXXFLAGS = -O3 -std=c++17 -I/usr/include/eigen3 -Wall -fopenmp

SRCDIR   = src
BINDIR   = bin

$(BINDIR):
	mkdir -p $(BINDIR)

# Fit library, C interface in sph_fit.h. Without OpenMP: it starts no threads.
$(BINDIR)/sph_fit.o: $(SRCDIR)/sph_fit.cpp $(SRCDIR)/sph_fit.h $(SRCDIR)/sph_harm_math.hpp | $(BINDIR)
	$(CXX) $(filter-out -fopenmp,$(CXXFLAGS)) -c -o $@ $(SRCDIR)/sph_fit.cpp

$(BINDIR)/libsphfit.a: $(BINDIR)/sph_fit.o
	$(AR) rcs $@ $<

# Production converter
$(BINDIR)/converter: $(SRCDIR)/converter.cpp $(SRCDIR)/sph_fit.h $(SRCDIR)/ft3_reader.hpp $(BINDIR)/libsphfit.a | $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCDIR)/converter.cpp $(BINDIR)/libsphfit.a

# Test of the C interface, compiled as C
$(BINDIR)/test_sph_fit: tests/test_sph_fit.c $(SRCDIR)/sph_fit.h $(BINDIR)/libsphfit.a | $(BINDIR)
	$(CC) -std=c99 -O2 -Wall -Wextra -I$(SRCDIR) -o $@ tests/test_sph_fit.c $(BINDIR)/libsphfit.a -lstdc++ -lm -pthread

# Validation binary (compare SH values against Python reference)
$(BINDIR)/validate_sph: $(SRCDIR)/validate_sph.cpp $(SRCDIR)/sph_harm_math.hpp | $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCDIR)/validate_sph.cpp

all: $(BINDIR)/converter $(BINDIR)/libsphfit.a $(BINDIR)/validate_sph

test: $(BINDIR)/test_sph_fit
	$(BINDIR)/test_sph_fit

clean:
	rm -rf $(BINDIR)

.PHONY: all test clean
