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

DEBUG=no # commento out if you want debugging information in the compilation
MAKEFILEH_DIR=../../
#
include $(MAKEFILEH_DIR)/Makefile.inc
#
# You may have an include file also in the current directory
#
-include Makefile.inc
#CPPFLAGS+=-DVERBOSE
#
# The general setting is as follows:
# mains are identified bt main_XX.cpp
# all other files are XX.cpp
#

# get all files *.cpp
SRCS=$(wildcard *.cpp)
# get the corresponding object file
OBJS = $(SRCS:.cpp=.o)
OBJS := $(filter-out test_gradientAD.o, $(OBJS))
LIB_OBJS = $(filter-out main%.o, $(OBJS))
# get all headers in the working directory
HEADERS=$(wildcard *.hpp)
#
exe_sources=$(filter main%.cpp,$(SRCS))
EXEC=$(exe_sources:.cpp=)

#========================== ORA LA DEFINIZIONE DEGLI OBIETTIVI
.phony= all clean distclean doc

.DEFAULT_GOAL = all

all: $(DEPEND) $(EXEC) test_gradientAD

clean:
	$(RM) -f $(EXEC) $(OBJS) test_gradientAD test_gradientAD.o

distclean:
	$(MAKE) clean
	$(RM) -f ./doc $(DEPEND)
	$(RM) *.out *.bak *~

doc:
	doxygen $(DOXYFILE)

$(EXEC): $(OBJS)

library: lineSearch.so

lineSearch.so:
	$(MAKE) DEBUG=no CXXFLAGS="$(CXXFLAGS) -fPIC"  $(LIB_OBJS)
	$(CXX) -shared $(LDFLAGS)  -o $@ $(LIB_OBJS)

test_gradientAD.o:  test_gradientAD.cpp GradientsAD.hpp

test_gradientAD: test_gradientAD.o

install: lineSearch.so
# install all header files in $(PACS_ROOT)/include
	cp $(HEADERS) $(PACS_ROOT)/include
# install the library in $(PACS_ROOT)/lib
	cp lineSearch.so $(PACS_ROOT)/lib

$(DEPEND): $(SRCS)
	$(RM) $(DEPEND)
	for f in $(SRCS); do \
	$(CXX) $(STDFLAGS) $(CPPFLAGS) -MM $$f >> $(DEPEND); \
	done

-include $(DEPEND)




