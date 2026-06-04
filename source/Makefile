include makefile.include

OBJECTS = numbers.o symmetry.o groups.o constraints.o clusters.o orbits.o tensors.o structure.o spaces.o parameter.o forcefields.o builder.o reader.o main.o

forcefield: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LIBS) -o $@
	mv $@ ../bin/.

lib: $(OBJECTS)
	$(CXX) $(CXXFLAGS_LIB) _forcefieldpy.cpp $(OBJECTS) $(LIBS) -o forceModelPy.so
	mv forceModelPy.so ../forcefield_ml/.

main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(INCS) -c main.cpp

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) $(INCS) -c $<

clean:
	rm -f *.o
