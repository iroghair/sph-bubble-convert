CXX      = g++
CXXFLAGS = -O3 -std=c++17 -I/usr/include/eigen3 -Wall -fopenmp

SRCDIR   = src
BINDIR   = bin

$(BINDIR):
	mkdir -p $(BINDIR)

# Production converter
$(BINDIR)/converter: $(SRCDIR)/converter.cpp $(SRCDIR)/sph_harm_math.hpp $(SRCDIR)/ft3_reader.hpp | $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCDIR)/converter.cpp

# Validation binary (compare SH values against Python reference)
$(BINDIR)/validate_sph: $(SRCDIR)/validate_sph.cpp $(SRCDIR)/sph_harm_math.hpp | $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRCDIR)/validate_sph.cpp

all: $(BINDIR)/converter $(BINDIR)/validate_sph

clean:
	rm -rf $(BINDIR)

.PHONY: all clean
