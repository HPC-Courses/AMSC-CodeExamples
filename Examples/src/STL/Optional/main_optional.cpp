#include "readOptional.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include <map>

//! function templates that counts valid optionals in a vector an prints the
//! valid values
//!
//! \tparam V a std::vector or std::array of optionals
//! \param v the passed container
//!
template <class V>
void
countValid(V const &v)
{
  int count{0};
  // A possible use of for_each
  // I exploit the fact that std::vector and std::array contain a type
  // set to the type of the contained objects and that an optional<T>
  // contextually converts to true if set, to false if unset
  std::for_each(v.begin(), v.end(), [&count](typename V::value_type const &x) {
    if(x)
      ++count;
  });

  std::cout << "The container has " << count << " valid elements:\n";
  for(auto &x : v)
    if(x) // valid
      {
        // Dereferencing an optional returns underlying value
        std::cout << *x << " ";
      }
  std::cout << std::endl;
}
std::string content(std::optional<double> const &x)
{
  // an example of use of or_else and value_or  and_then
  // if values is set string contains "set" otherwise "unset"
  // This can be done with just an if, ut is a nice example 
  // of how to use the new methods of std::optional
  std::string s = x.and_then([]([[maybe_unused]] double const & v) {
                      return std::optional<std::string>("set");
                    })
                      .value_or("unset");
  return s;
}
int
main()
{
  // A vector of 10 optionals
  // The optional are default contructed and thus are empty (i.e. invalid)
  std::vector<std::optional<double>> v(10);

  v[0] = 10.0;
  v[3] = 30.0;
  countValid(v);
  // another way of thesting if a value is set, the method has_value,
  // and to get the value: value() The latter throws an exception
  // if the value is not set, so it is safer than *
  if(v[2].has_value())
    std::cout << "v[2] stores " << v[2].value() << std::endl;
  else
    std::cout << "v[2] stores no value" << std::endl;
  v[2] = 20.0;
  countValid(v);

  // You may also exploit the fact that a std::optional<T> converts to bool
  if(v[2])
    std::cout << "v[2] stores " << v[2].value() << std::endl;
  else
    std::cout << "v[2] stores no value" << std::endl;
  // You can reset a value to "unset"
  v[2].reset();
  if(v[2].has_value())
    {
      std::cout << "v[2] stores " << v[2].value() << std::endl;
    }
  else
    {
      std::cout << "v[2] stores no value" << std::endl;
    }

  std::cout << " Now reading from file\n";
  using namespace apsc::optionalExample;
  std::ifstream file("data.txt");
  auto          values = apsc::optionalExample::readValues(file);
  for (int i=0; auto const &x : values)
    {
      std::cout << "value " << i << " is " << content(x) << "\n";
      ++i;
    }
  std::cout << values;
 // extract the set values in values and insert them in a map<std::size_t,double> containing the index and 
// the value. I use  and_then to do this in a single statement. Note that and_then must return 
// an optional, which here is a dummy optional<bool> that is not used. 
// The lambda captures the map by reference and the index by value.

  std::map<std::size_t,double> m;
  for (std::size_t i=0; i<values.size(); ++i)
    {
      values[i].and_then([&m,i](double const &v) { m[i] = v; return std::optional<bool>{true}; });
    }
  std::cout << "The map contains:\n";
  for (auto const &p : m)
    {
      std::cout << "index " << p.first << " value " << p.second << "\n";
    }
}
