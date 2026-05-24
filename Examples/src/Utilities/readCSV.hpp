#ifndef HH_READCVSHEADER_HH
#define HH_READCVSHEADER_HH
#include "string_utility.hpp"
#include <iostream>
#include <string>
#include <vector>
namespace Utility
{
/*!
 * Class for reading a CSV file. It supports quoted text cells, so separator
 * characters inside `"..."` are treated as part of the cell content.
 */
class ReadCSV
{
public:
  //! The type used for characters
  using CharT = char;
  //! The type used for strings
  using StringType = std::string;
  //! The type used for the Token, i.e. the content of a cell
  using Token = StringType;
  //! The type used to store a record
  using Record = std::vector<Token>;
  //! The type used to store all records
  using RecordList = std::vector<Record>;
  ReadCSV() = default;
  /*!
   * Change the separator character
   * The default is a comma.
   */
  void
  setSeparator(CharT s = CharT{','})
  {
    sep = s;
  }

  /*
   * When reading a token protected by double quotation marks (" ") strip the
   * marks from the stored tokens (default: `false`, so quotes are kept).
   */
  void
  stripQuotation(bool strip = false)
  {
    stripApexes = strip;
  }
  /*!
   * Change the minimum number of tokens in a record (default: 1).
   * If we read a number of tokens less than the minimum, the record is
   * completed with empty cells.
   * If different from 1, it is also used to preallocate memory.
   */
  void
  setMinTokens(unsigned n = 1u)
  {
    minTokens = n;
  }
  /*!
   * Change the minimum number of records
   * If different from 0, it is used to preallocate memory.
   */
  void
  setMinRecords(unsigned n = 0u)
  {
    minRecords = n;
  }
  //! Change the number of lines to be skipped (default 0)
  void
  setSkippedLines(unsigned s = 0u)
  {
    skip = s;
  }
  //! Change verbosity (default false)
  void
  setVerbose(bool v = false)
  {
    verbose = v;
  }
  /*!
   * Get the parsed records. It returns a matrix
   * so that using `[i][j]` you access the `j`-th token of record `i`
   * (remember, starting from 0).
   * @return The matrix containing all records and tokens.
   */
  RecordList
  getTokens()
  {
    return allRecords;
  }
  //! Clear all records and tokens
  void
  clear()
  {
    allRecords.clear();
    allRecords.shrink_to_fit();
  }
  /*!
   * Read all tokens from an input stream.
   * @param in The input stream.
   */
  void read(std::basic_istream<CharT> &in);
  //! Writes all records to an output stream.
  void writeAllRecords(std::basic_ostream<CharT> &) const;

private:
  CharT                  sep = CharT{','};
  unsigned int           minTokens = 1u;
  unsigned int           skip = 0u;
  bool                   verbose = false;
  RecordList             allRecords;
  unsigned int           minRecords = 0u;
  bool                   stripApexes = false;
  static constexpr CharT textMark = CharT{'"'};
  void writeRecord(std::basic_ostream<CharT> &, Record const &) const;
};

} // namespace Utility

#endif
