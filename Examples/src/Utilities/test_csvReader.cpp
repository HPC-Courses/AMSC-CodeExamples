/*
 * test_csvReader.cpp
 *
 *  Created on: Jun 24, 2021
 *      Author: forma
 */

#include "readCSV.hpp"
#include <fstream>
#include <iostream>
int
main()
{
  using namespace Utility;
  std::ifstream db("test.csv");
  ReadCSV       reader;
  // Every record has at least 3 tokens
  reader.setMinTokens(3u);
  // Set the minimum number of records to 4.
  // If you know the number of records, this avoids unnecessary reallocations.
  reader.setMinRecords(4u);
  // Enable verbose output.
  reader.setVerbose(true);
  // Skip the first line.
  reader.setSkippedLines(1);
  // Strip quotation marks from text tokens, e.g. "anc" -> anc.
  reader.stripQuotation(true);
  // Read the file.
  reader.read(db);
  // Write the parsed records.
  reader.writeAllRecords(std::cout);
}
