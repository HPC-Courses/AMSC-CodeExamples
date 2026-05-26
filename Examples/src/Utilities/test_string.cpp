#include "string_utility.hpp"
#include <iostream>
#include <vector>
int
main()
{
  using namespace std;
  string a("  I have spaces at my left and at my right ");
  string a_copy(a);
  cout << "Original     :#" << a << "# string is between hashes" << endl;
  cout << "Trimmed  left:#" << Utility::ltrim(a) << "#" << endl;
  // `ltrim()` returns a trimmed copy; it does not modify `a`.
  cout << "Trimmed right:#" << Utility::rtrim(a) << "#" << endl;
  cout << "Trimmed fully:#" << Utility::trim(a) << "#" << endl;
  cout << "Upper Case   :" << Utility::toupper(a) << endl;
  cout << "lower case   :" << Utility::tolower(a) << endl;
  std::cout
    << "Reading a whole file into a string stream unsing GlobbedTextReader\n";
  Utility::GlobbedTextReader globbedText("test_string.cpp");
  std::stringstream         &sstream = globbedText.globbedText();
  std::cout << "the stream contains " << sstream.str().size()
            << " characters\n";
  // check if read was successful
  std::ifstream testfile("test_string.cpp");
  bool          ok = true;
  while(!sstream.eof() && !sstream.bad())
    {
      std::string line;
      std::string lineoriginal;
      std::getline(sstream, line);
      std::getline(testfile, lineoriginal);
      if(line != lineoriginal)
        {
          std::cerr << "Error: line read from stream does not match line read "
                       "from file\n";
          ok = false;
          break;
        }
    }
  testfile.close();
  if(ok)
    std::cout << "File read successfully\n";
  // Now extract the lines I have to reset the stream to read from the beginning
  sstream.clear(); // clear flags in case of problems

  sstream.seekg(0, std::ios::beg);
  // for some reasons I need to reset the sting stream state as well

  auto lines = Utility::chop(sstream);
  std::cout << "the stream contains " << lines.size() << " lines\n";
  sstream.str(""); // clear buffer

  // testing readWholeFile
  std::cout << "Reading a whole stream into a string unsing readWholeFile\n";
  std::ifstream myfile("test_string.cpp", std::ios::in | std::ios::binary);
  if(!myfile.is_open())
    {
      myfile.close();
      throw std::runtime_error("Cannot open file test_string.cpp");
    }
  // I put the string in a sting stream to check that the content is the same as
  // the original file
  auto globbedfile = std::istringstream(Utility::readWholeFile(myfile));
  std::cout << "The file has " << globbedfile.str().size() << " characters\n";
  myfile.close();
  testfile.open("test_string.cpp");
  ok = true;
  while(!globbedfile.eof() && !globbedfile.bad())
    {
      std::string line;
      std::string lineoriginal;
      std::getline(globbedfile, line);
      std::getline(testfile, lineoriginal);
      if(line != lineoriginal)
        {
          std::cerr << "Error: line read from stream does not match line read "
                       "from file\n";
          ok = false;
          break;
        }
    }
  testfile.close();
  if(ok)
    std::cout << "File read successfully\n";
  std::cout << " Now testing string distance\n";
  std::vector<std::string> v1{"Luca", "John", "cat", "plain"};
  std::vector<std::string> v2{"Lucia", "Mary", "cut", "plane"};
  std::cout << "String1\tString2\tDistance\n";
  for(auto i = 0u; i < v1.size(); ++i)
    {
      std::cout << v1[i] << "\t" << v2[i] << "\t"
                << Utility::stringDistance(v1[i], v2[i]) << std::endl;
    }
}
