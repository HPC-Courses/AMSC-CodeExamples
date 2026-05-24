# Various Utilities

This directory contains utilities that can be included and used in
other examples. You should therefore install them before moving on to the
other examples.

Run:

    make 
    make install

to install them. `make` produces both a shared and a static library,
called `libpacs.so` and `libpacs.a`, respectively. `make install`
installs the header files into `PACS_ROOT/include` and the libraries into
`PACS_ROOT/lib`. Use `make DEBUG=no` if you want the library code to be
optimized.

After installing the libraries, you can run:

    make test

This builds tests for most of the utilities. All test executables start
with `test_`. You may inspect the source files to see what they do.

**Remember to install the utilities, they are used by other examples!**


**Note**: some utilities live in a nested namespace inside `apsc`. Check
the code or the examples to see the exact layout.

List of the utilities:

* `absdiff` If you need the absolute difference of two integral values, you
  cannot simply use `std::abs(x-y)`: if the types are unsigned, the result is
  wrong when `x > y`. This utility computes the correct result when both
  arguments are signed or both are unsigned. It rejects mixed cases because the
  intent would be ambiguous.

* `Arithmetic.hpp` introduces concepts to constrain template types to floating-
  point or integral types, optionally including complex numbers.

* `booleanConcept.hpp` A concept expressing the semantics of a boolean type.

* `chrono` A timing utility built on top of the Standard Library `chrono`
  facilities.

* `CloningUtilities` Tools for clonable classes (Prototype design pattern). It
  contains type traits to test whether a class `T` provides the following
  (usually virtual) member function:

```cpp
std::unique_ptr<B> clone() const;
```

that returns a pointer to a copy of the object wrapped in a
`unique_ptr`. `B` can be either `T` or a base class of `T`. It also
contains an interesting class, `PointerWrapper`, which implements an owning
pointer with deep-copy semantics. The pointed-to class must be clonable. This
lets you implement composition with a polymorphic object and obtain the copy
operations of the composing class automatically.

* `cxxversion` Reports which C++ standard version is being used to compile your
  code.

* `extendedAssert` Assertions with messages. It extends the `assert` macro so
  that you can attach a message. There are also switches that can be enabled
  with `-DXXX` compiler options to change the behavior of some checks.

* `Factory`  A generic object factory. Inspired by a code by [Andrei Alexandrescu](https://en.wikipedia.org/wiki/Andrei_Alexandrescu). 

* `GetPot` GetPot command parser <http://getpot.sourceforge.net/>. The version
  here has been simplified, so you only need `#include <GetPot>` or
  `#include "GetPot"` to use it.

* `gnuplot-iostream` A stream interface to open `gnuplot` from within a
  program. Useful for simple visualizations. You need
  [gnuplot](http://www.gnuplot.info/) installed on your system (it is available
  as a Debian package).

* `hashCombine.hpp` Provides the function object `hash_combine`, which can be
  used to combine the hash keys of objects of different types, provided they
  have `std::hash` defined. It can be used to build the hash key of a user-
  defined class by combining the hashes of its non-static data members, in
  order to obtain better uniformity. Usage is explained in the file.

* `is_complex.hpp` A header containing a type trait that checks whether a type
  is `std::complex<T>`. It also defines the concept `Complex`, used to
  constrain a template parameter to `std::complex<T>`, and
  `ArithmeticComplex`, which further requires `T` to be either floating-point
  or integral.

* `is_eigen.hpp` A header containing type traits to test whether a type is an
  `Eigen::Matrix`. It also provides traits and concepts to distinguish sparse
  and dense matrices.

* `is_specialization.hpp`. Type traits and concepts to test if a class is the specialization of a class template.

* `JoinVectors.hpp` An example showing how to mimic Python's `zip`. You can use
  it to iterate jointly over a set of vectors.
  
* `overloaded` A facility, called `overloaded` that implements the overloaded design pattern that may be used to visit a `std::variant`.

* `parallel_for` An example of metaprogramming used to implement a parallel
  `for` loop. It also shows some uses of concepts.

* `Proxy.hpp` Despite the name, it is not a proxy. It is a utility that can be
  used to register objects in an object factory automatically.

* `range_to_vector` If you create a range view, for example with
  `std::views::iota` or by applying views to a vector, you cannot directly use
  it to initialize a `std::vector`. A proposal exists for a future C++
  standard, but for now we still need to do it ourselves. This utility
  converts a range to a vector. It is a simple wrapper around
  `std::ranges::copy`. More information can be found
  [here](https://timur.audio/how-to-make-a-container-from-a-C++20-range).

* `readCSV` A class for reading CSV files. Useful if you have data in a
  spreadsheet and want to load it into a C++ program. Better tools exist, but
  this one is relatively simple and handy.

* `scientific_precision` A function that sets the precision of a stream to the maximum value for a floating point. It contains also stream manipulators for the same purpose.

* `setUtilities` Three utilities that simplify set operations
  (union/difference/intersection) on ordered containers. They are built on top
  of the corresponding Standard Library algorithms, but expose a simpler
  interface.

* `StatisticsComputations.hpp` Some tools to compute basic statistics of a sample.

* `string_utility` Extra string utilities: trimming, lower/upper-case
  conversion, reading a whole text file into a buffer (faster, but potentially
  memory-hungry), and computing the Levenshtein edit distance between two
  strings.

* `toString` Converts anything that supports the `<<` streaming operator into a
  string. It is a straightforward use of `std::stringstream`.

* `tuple_utilities.hpp` Contains some utilities for tuples:  `tuple_common_type_t<Tuple>` that returns the common type of all types contained in a tuple, and `for_each<Tuple F>` and `for_each2<Tuple, F>` that apply (possibly in parallel) the function object `F` to all elements of the tuple. The first one returns a tuple with the result, the second one does not and is thus applicable also if `F` is a void function. `all_of<Tuple,F>` and `any_of<Tuple,F>`, that apply predicate `F` to all elements of a tuple. The first returns true if the predicate is true for all elements, the second if it is true for at least one element.

* `type_name.hpp` A utility to pretty-print the type name of a variable. It is
  useful for debugging. It is based on `boost::core::demangle`.
   
**Note**: `Factory.hpp` and `Proxy.hpp` are actually links to the same file in
the `GenericFactory` folder. If those files are missing for some reason, you
may safely copy them from `GenericFactory/` into `Utilities/`.


## An explanation of `hash_combine`.##
The functor `apsc::hash_combine` provided by  `hashCombine.hpp` is given in two formats, one used *fold expressions* introduced in C++17, the other requires just C++11 to work. Since they are short but complicated it is worthwhile giving some explanation.
We recall that a good hash function should satisfy as far as possible the uniformity property: the probability distribution of the hash keys (conditioned to the distribution of the possible arguments) should be uniform. Since we do not normally know the distribution of the arguments, it is normally assumed that it is uniform as well.
This is not so easy to achieve and a badly unbalanced hash function may make your unordered container very inefficient! That's why a lot of "tricks" are used to increase entropy and avoid "clustering".

Here the C++=17 version of my `hash_combine` (taken from the web, I thank the unknown author):

	template <typename T, typename... Rest>
	void hash_combine(std::size_t& seed, const T& v, const Rest&... rest)
	{
	   std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b97f4a7c15 + (seed << 6) + (seed >> 2);
        (hash_combine(seed,rest), ...);
	}

Note that **`seed` is passed by reference**, and indeed it will finally contain the hash key! See the documentation in `hashCombine.hpp` or `test_hash_combine.cpp` for the way to use this utility for the construction of the hash function for your class.

We use some unusual operators: `operator^()` is a binary operator that performs a bit-wise exclusive or (xor). That is if we have two integers, `a` and `b`, whose binary representation is `a=0b1001` and `b=0b1100`, we have `a^b=0b0101` (The `0b` indicates that what follows is a binary literal). You can compare with the result of the two other binary bit-wise logical operators: `a|b=0b1101` (bit-wise or) and `a&b=0b1010` (bit-wise and). In this context, bit-wise xor is used to introduce a bit of entropy. 
Then, we add the "magic number" `0x9e3779b9` (`0x` indicates that is is in hexadecimal format). It is the integral part of the Golden Ratio's fractional part `0.61803398875…`  multiplied by `2^64`. Adding it has a scattering effect, often referred to as "Golden Ratio Hashing", or "Fibonacci Hashing" and was popularised by Donald Knuth (The Art of Computer Programming: Volume 3: Sorting and Searching). If you are interested, in number theoretical terms it is related to the Steinhaus Conjecture. 

After doing that, we have the `<<` and `>>` operators. These are the left and right *bit-shift* operators that take a number to shift and an unsigned integer indicating the number of shifts. To understand how it works let assume that the variable `a=0b1001` and `b` defined above is just a 4-bit integer (to make things simpler). We have
`(a<< 1)=0b0010`, `(a<< 2)=0b0100`, and `(a>>1)=0100`, `(a>>2)=0010`. I hope it is clear. Again, all this fuss is to scatter the digits around (in fact the bits).

Finally, I am using here a fold expression in `(hash_combine(seed,rest), ...);` to expand the variadic template. It means that, for intance,
`hash_combine(0,a,b,C++)` expands in

	std::hash<T> hasher;
	seed ^= hasher(a) + 0x9e3779b97f4a7c15 + (seed << 6) + (seed >> 2);
	hash_combine(seed,b);
	hash_combine(seed,C++);

thanks to the magic of a fold expression. The pre-C++17 version achieves fold expression with a dirty trick that I avoid explaining (after all, C++17 is now well established).

## What you can learn from these examples##

The files in this directory illustrate the advantage of having little general utilities that can be integrated in different codes.
Some utilities are simple to understand, like `Chrono` or `StatisticsComputations`, others make use of more sophisticated generic programming 
techniques, like `Factory`, or template metaprogramming, like `is_complex`, `is_eigen`, `joinVectors` and `CloningUtilities`. 
Finally, some, like `gnuplot_iostream` and `GetPot` are just copies of tools available open source, copied here for simplicity.

In `CloningUtilities` and `joinVectors` you have classes that define a dereferencing operator (`*`) and an access via pointer operator
(`->`). Something you do not find very often.

In `hashCombine.hpp` the use of some bit-wise operators and fold expression.

