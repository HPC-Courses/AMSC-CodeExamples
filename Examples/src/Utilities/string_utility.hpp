#ifndef HHH_STRING_UTILITIES
#define HHH_STRING_UTILITIES
#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
/*
  Part of this software was adapted from code found online. I thank the
  original, unknown authors.
  It has been revised to improve readability using C++11 features, even if
  that may make it slightly less efficient than the original code.
 */
namespace Utility
{
//! Trims leading whitespace from a string.
/**
   @param sin The input string.
   @return A copy of the string with leading whitespace removed.
 */
inline std::string
ltrim(std::string sin)
{
  auto        s = std::move(sin);
  auto const &loc = std::locale();
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                  [&loc](std::string::const_reference c) {
                                    return !std::isspace(c, loc);
                                  }));
  return s;
}

//! Trims trailing whitespace from a string.
/**
   @param sin The input string.
   @return A copy of the string with trailing whitespace removed.
 */
inline std::string
rtrim(std::string sin)
{
  auto        s = std::move(sin);
  auto const &loc = std::locale();
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [&loc](std::string::const_reference c) {
                         return !std::isspace(c, loc);
                       })
            .base(),
          s.end());
  return s;
}

//! Trims whitespace from both ends of a string.
/**
   @param s The input string.
   @return A copy of the string with leading and trailing whitespace removed.
 */
inline std::string
trim(std::string s)
{
  return ltrim(rtrim(s));
}

//! Converts a whole string using the current locale.
/**
   @param s A string.
   @return The modified string.
 */

/**@{*/
std::string toupper(std::string const &s);
std::string tolower(std::string const &s);
/**@}*/

/*!
 * A functor for case-insensitive string comparison.
 */
struct compareNoCase
{
  /*!
   * @param a First string.
   * @param b Second string.
   * @return The result of comparing the strings case-insensitively.
   */
  inline bool
  operator()(std::string const &a, std::string const &b) const
  {
    return Utility::toupper(a) < Utility::toupper(b);
  }
};

/*!
 * Reads a whole line and returns it as an input string stream.
 * @param stream An input stream.
 * @return An input string stream containing the extracted line.
 */
std::istringstream nextLine(std::istream &stream);

/*!
 * When consuming whitespace-delimited input (e.g. int n; std::cin >> n;) any
 * whitespace that follows, including a newline character, will be left on the
 * input stream. Then, when switching to line-oriented input, the first line
 * retrieved with getline() will be just that whitespace. In the likely case
 * that this is unwanted behaviour, one possible solution is to call this
 * function on the stream before switching to `getline()`.
 *
 * @param istream The input stream.
 */
void cleanStream(std::istream &istream);
/*!
 * Helper class that manages a string stream backed by a whole text file loaded
 * into memory.
 *
 * Reading can be faster because the file is loaded in a single block, at the
 * cost of higher memory usage.
 *
 * Copy operations are not provided (but move operations are available) because
 * the class is meant only as a helper and copy semantics would be ambiguous:
 * should the text buffer be copied deeply or shared?
 *
 * You can extract the data through the streaming operator or by accessing the
 * underlying string stream by reference.
 */
class GlobbedTextReader
{
public:
  /*!
   * Default constructor.
   */
  GlobbedTextReader() = default;
  /*!
   * Builds the object and immediately reads a text file.
   * @param fileName The file name.
   * @throw std::runtime_error if the file cannot be opened.
   */
  GlobbedTextReader(std::string const &fileName);
  /*!
   * Reads the entire contents of a text file.
   * @param fileName The file name.
   * @throw std::runtime_error if the file cannot be opened.
   */
  void read(std::string const &fileName);
  /*!
   * Extracts data from the stored text using the standard streaming operator.
   *
   * @tparam T The data type.
   * @param data The destination object.
   * @return This object.
   */
  template <class T> GlobbedTextReader &operator>>(T &data);
  /*!
   * Returns the enclosed string stream for direct access.
   * @note you should extract it only by reference.
   * @return A const reference to the string stream associated with the text
   * buffer.
   */
  std::stringstream const &
  globbedText() const
  {
    return MyGlobbedText;
  }
  /*!
   * Returns the enclosed string stream for direct access.
   * @note You should access it only by reference.
   *
   * @return The string stream associated with the text buffer.
   */
  std::stringstream &
  globbedText()
  {
    return MyGlobbedText;
  }
  /*!
   * Releases the buffer.
   *
   * This function exists only to release memory after you are done using the
   * string stream. After calling `close()`, the object should not be used
   * again.
   */
  void
  close()
  {
    MyGlobbedText.str({});
    MyGlobbedText.clear();
    MyBuffer.reset();
    MySize = 0;
  }
  /*!
   * @return The size of the internal buffer, in bytes.
   */
  std::size_t
  size() const
  {
    return MySize;
  }

private:
  //! The underlying string stream.
  std::stringstream MyGlobbedText;
  //! The buffer that stores the file contents.
  std::unique_ptr<char[]> MyBuffer;
  //! Size of the internal buffer.
  std::size_t MySize = 0;
  //! Returns the internal raw buffer pointer.
  char *
  buffer()
  {
    return MyBuffer.get();
  }
  //! Resets reading to the beginning of the stream.
  void
  setAtStart()
  {
    MyGlobbedText.seekg(0, std::ios::beg);
  }
};

template <class T>
GlobbedTextReader &
GlobbedTextReader::operator>>(T &data)
{
  this->MyGlobbedText >> data;
  return *this;
}

/*!
@brief Reads a whole file into a string.
@details This is a simpler alternative to `GlobbedTextReader`. It uses the
Scott Meyers idiom. It may be less efficient than `GlobbedTextReader`, but it
is simpler and often easier to use when you just need the whole file as a
string.
@param file The file stream from which to read. It must already be open and
ready for reading.
@return The whole file as a string.
*/
inline std::string
readWholeFile(std::istream &file)
{
  return std::string{std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>()};
}

/*!
 * Splits a string stream into a vector containing one string per line.
 *
 * @param sstream The string stream. It must be in a valid state.
 * @return A vector containing the extracted lines.
 */
std::vector<std::string> chop(std::stringstream &sstream);

/*!
Compute the Levenshtein edit distance between two strings.

Levenshtein distance is the number of edits needed to transform one
string into the other. If it is zero, the strings are identical.

Readapted from code by Jonathan Wood, found at
http://www.blackbeltcoder.com/

@note This function should be used only with small strings. It
builds a matrix internally (as a vector of vectors) whose size is the
product of `(a.size() + 1) * (b.size() + 1)`. If you need a more memory-
efficient algorithm, note that in practice only a band of the matrix is
required, but the implementation becomes more complicated. The advantage of
this version is its simplicity.

@param a First string.
@param b Second string.
@return The Levenshtein distance. Larger values mean the strings are more
different.

*/
unsigned int stringDistance(std::string const &a, std::string const &b);
} // namespace Utility
#endif
