############################################################
#
# An example of Makefile for the course on 
# Advanced Programming for Scientific Computing
# It should be modified for adapting it to the various examples
#
############################################################
#
# The environmental variable PACS_ROOT should be set to the
# root directory where the examples reside. In practice, the directory
# where this file is found. The resolution of PACS_ROOT is made in the
# Makefile.h file, where other important variables are also set.
# The only user defined variable that must be set in this file is
# the one indicating where Makefile.h resides

MAKEFILEH_DIR=../../../../
#
DEBUG=NO
include $(MAKEFILEH_DIR)/Makefile.inc
#
# You may have an include file also in the current directory
#
-include Makefile.inc

# Keep OpenMP flags out of CXX: normal builds use MPI only.
OPENMP_FLAGS ?= -fopenmp
# GetPot is supplied by the repository's Utilities directory.
CPPFLAGS += -I../../../Utilities
#
# The general setting is as follows:
# mains are identified bt main_XX.cpp
# all other files are XX.cpp
#
# get all files *.cpp
SRCS=$(wildcard *.cpp)
# get the corresponding object file
OBJS = $(SRCS:.cpp=.o)
# get all headers in the working directory
HEADERS=$(wildcard *.hpp)
#
exe_sources=$(filter main%.cpp,$(SRCS))
EXEC=$(exe_sources:.cpp=)
OPENMP_OBJS=$(SRCS:.cpp=.openmp.o)
OPENMP_EXEC=$(addsuffix _openmp,$(EXEC))

#========================== ORA LA DEFINIZIONE DEGLI OBIETTIVI
.PHONY: all openmp clean distclean doc

.DEFAULT_GOAL = all

all: $(DEPEND) $(EXEC)

# Separate objects and executables allow both build modes to coexist.
openmp: $(DEPEND) $(OPENMP_EXEC)

$(OPENMP_EXEC): $(OPENMP_OBJS)
	$(CXX) $(LDFLAGS) $(TARGET_ARCH) $(OPENMP_FLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@

%.openmp.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP_FLAGS) -c $< -o $@

clean:
	$(RM) -f $(EXEC) $(OBJS) $(OPENMP_EXEC) $(OPENMP_OBJS)

distclean:
	$(MAKE) clean
	$(RM) -f ./doc $(DEPEND)
	$(RM) *.out *.bak *~

doc:
	doxygen $(DOXYFILE)

$(EXEC): $(OBJS)

$(OBJS): $(SRCS)

# Rebuild existing objects when the build configuration changes.
$(OBJS) $(OPENMP_OBJS): Makefile Makefile.inc $(MAKEFILEH_DIR)/Makefile.inc

$(DEPEND): $(SRCS) Makefile Makefile.inc $(MAKEFILEH_DIR)/Makefile.inc
	$(RM) $(DEPEND)
	for f in $(SRCS); do \
	$(CXX) $(STDFLAGS) $(CPPFLAGS) -MM \
	  -MT "$${f%.cpp}.o" -MT "$${f%.cpp}.openmp.o" $$f >> $(DEPEND); \
	done

-include $(DEPEND)
