/*******************************************************************************************/
/* SQL Programming Demo - FAO Crop and Livestock Production                                */
/*                                                                                         */
/* Author: Michael J. Bernier                                                              */
/*                                                                                         */
/* This project contains a series of queries written in Microsoft T-SQL to demonstrate my  */
/* skills in SQL programming.                                                              */
/*                                                                                         */
/* Tools Utilized:                                                                         */
/*                                                                                         */
/* - Microsoft SQL Server 2019 Developer Edition                                           */
/* - SQL Server Management Studio v19.1                                                    */
/*                                                                                         */
/* About the data:                                                                         */
/*                                                                                         */
/* The dataset used in this project is sourced from the Food and Agricultural              */
/* Organization (FAO) of the United Nations, and consists of crop and livestock            */
/* production data by country and region from 1961 to 2021. This data contains statistics  */
/* for 307 raw and processed agricultural products including grains, vegetables, fruits,   */
/* nuts, meats, and dairy.                                                                 */
/*                                                                                         */
/* Website: https://fao.org/faostat/                                                       */
/*                                                                                         */
/* NOTE: The original structure of the Production_Data table contained overlapping columns */
/* of detail also provided in the Production_AreaCodes and Production_ItemCodes tables.    */
/* These overlaps have been removed from the Production_Data table. I also found that      */
/* the Element values could be removed into their own separate table as well, now called   */
/* Production_ElementCodes. In addition, I found it helpful to also create another         */
/* dimension table to describe the units of measure listed in the data; that table is      */
/* called Production_Units. These changes do not materially affect the original intent,    */
/* but rather restructure it to better fit into a relational model/star schema.            */
/*                                                                                         */
/* The tables included in this database/demo are:                                          */
/* Production_AreaCodes - a dimensional table identifying each country or region           */
/* Production_Data - a fact table identifying each year/country/product/processing stage   */
/* Production_ElementCodes - a dimensional table identifying each processing stage         */
/* Production_Flags - a dimentional table identifying the source/accuracy of each fact     */
/* Production_ItemCodes - a dimensional table identifying each product                     */
/* Production_Units - a dimensional table identifying each unit of measure                 */
/*                                                                                         */
/*******************************************************************************************/

-- Housekeeping: set up environment and clear out leftover objects

USE FAO_Production;
GO

DROP TABLE IF EXISTS [tempdb].[dbo].[Reccount];
DROP TABLE IF EXISTS [tempdb].[dbo].[Tablelist];
DROP FUNCTION IF EXISTS [dbo].[Select_Area_and_Item];
GO


/*******************************************************************************************/
/* PART 1                                                                                  */
/*******************************************************************************************/
/* Display record counts and samples of each table in the database to obtain a basic       */
/* understanding of their structure and content                                            */
/*******************************************************************************************/


-- Set up variables

DECLARE @TABLENAME NVARCHAR(128) = '';
DECLARE @SQL NVARCHAR(MAX) = '';
DECLARE @LOOPCOUNT INT = 0;


-- Build a temp table containing a list of the tables in the database

SELECT TABLE_NAME INTO [tempdb].[dbo].[Tablelist] FROM INFORMATION_SCHEMA.TABLES;


-- Use a cursor to loop through the table list and build dynamic SQL statements to count
-- the records and write the values to another temp table

DECLARE TABLE_CURSOR CURSOR FOR
	SELECT * FROM [tempdb].[dbo].[Tablelist];
	OPEN TABLE_CURSOR
	FETCH NEXT FROM TABLE_CURSOR INTO @TABLENAME
	WHILE @@FETCH_STATUS = 0
		BEGIN
			IF @LOOPCOUNT = 0
				BEGIN
					SET @sql = 'SELECT CONVERT(NVARCHAR(128), N' + NCHAR(39) + @TABLENAME + NCHAR(39) + ') AS [Table Name], 
					COUNT(*) AS [Records] INTO [tempdb].[dbo].[Reccount] 
					FROM [FAO_Production].[dbo].[' + @TABLENAME + '];'
					EXEC(@sql);
					SET @LOOPCOUNT = 1
				END
			ELSE	
				BEGIN
					SET @sql = 'INSERT INTO [tempdb].[dbo].[Reccount] SELECT CONVERT(NVARCHAR(128), N' + NCHAR(39) + 
					@TABLENAME + NCHAR(39) + ') AS [Table Name], COUNT(*) AS [Records] 
					FROM [FAO_Production].[dbo].[' + @TABLENAME + '];'
					EXEC(@sql);
				END
			FETCH NEXT FROM Table_Cursor INTO @TABLENAME;
		END
	CLOSE TABLE_CURSOR;

-- Display the list of tables with record counts

SELECT * FROM [tempdb].[dbo].[Reccount];


-- Use the cursor to loop through the table list again and build dynamic SQL statements to
-- display the first 10 records in each database table

	OPEN TABLE_CURSOR
	FETCH NEXT FROM TABLE_CURSOR INTO @TABLENAME
	WHILE @@FETCH_STATUS = 0
		BEGIN
			SET @sql = 'SELECT TOP 10 ' + NCHAR(39) + @TABLENAME + NCHAR(39) + ' AS [Table Name],
			* FROM [FAO_Production].[dbo].[' + @TABLENAME + '];'
			EXEC(@sql);
			FETCH NEXT FROM Table_Cursor INTO @TABLENAME;
		END
	CLOSE TABLE_CURSOR;

DEALLOCATE TABLE_CURSOR;


/*******************************************************************************************/
/* PART 2                                                                                  */
/*******************************************************************************************/
/* Perform Exploratory Analysis of the data                                                */
/*******************************************************************************************/

-- Find the Item Code for apples

SELECT * FROM [dbo].[Production_ItemCodes]
WHERE [Item] LIKE '%apple%';

-- Apples are Item_Code 515


-- Find all the production entries for apples

SELECT * FROM [dbo].[Production_Data]
WHERE [Item_Code] = 515;


-- Find the Area Code for Canada

SELECT * FROM [dbo].[Production_AreaCodes]
WHERE [Area] LIKE '%Canada%';

-- Canada is Area Code 33


-- Find all the production entries for apples in Canada

SELECT * from [dbo].[Production_Data]
WHERE [Item_Code] = 515 and [Area_Code] = 33;


-- Add more descriptive info:
-- Join to the AreaCodes table to add the name of the Area to the list
-- Join to the ItemCodes table to add the name of the Item to the list
-- Join to the ElementCodes table to add the processing Element to the list

SELECT pd.*, ac.[Area], ic.[Item], ec.[Element]
FROM [dbo].[Production_Data] pd
LEFT JOIN [dbo].[Production_AreaCodes] ac on pd.[Area_Code] = ac.[Area_Code]
LEFT JOIN [dbo].[Production_ItemCodes] ic on pd.[Item_Code] = ic.[Item_Code]
LEFT JOIN [dbo].[Production_ElementCodes] ec on pd.Element_Code = ec.[Element_Code]
WHERE pd.[Item_Code] = 515 and pd.[Area_Code] = 33;


-- Re-sort the list by Area_Code and Year
-- Clean up the report by removing code columns and rearranging descriptive text
-- Also sort list by year and processing element

SELECT
	 ac.[Area]
	,ic.[Item]
	,pd.[Year]
	,ec.[Element]
	,pd.[Value]
	,pd.[Unit]
FROM [dbo].[Production_Data] pd
LEFT JOIN [dbo].[Production_AreaCodes] ac on pd.[Area_Code] = ac.[Area_Code]
LEFT JOIN [dbo].[Production_ItemCodes] ic on pd.[Item_Code] = ic.[Item_Code]
LEFT JOIN [dbo].[Production_ElementCodes] ec on pd.Element_Code = ec.[Element_Code]
WHERE pd.[Item_Code] = 515 and pd.[Area_Code] = 33
ORDER BY pd.[Year], ec.[Element];


-- Display the five years with the highest Yield

SELECT TOP(5)
	 ac.[Area]
	,ic.[Item]
	,pd.[Year]
	,ec.[Element]
	,pd.[Value]
	,pd.[Unit]
FROM [dbo].[Production_Data] pd
LEFT JOIN [dbo].[Production_AreaCodes] ac on pd.[Area_Code] = ac.[Area_Code]
LEFT JOIN [dbo].[Production_ItemCodes] ic on pd.[Item_Code] = ic.[Item_Code]
LEFT JOIN [dbo].[Production_ElementCodes] ec on pd.Element_Code = ec.[Element_Code]
WHERE pd.[Item_Code] = 515 and pd.[Area_Code] = 33 and ec.[Element] = 'Yield'
ORDER BY pd.[Value] DESC;


-- Display the five years with the lowest Area Harvested

SELECT TOP(5)
	 ac.[Area]
	,ic.[Item]
	,pd.[Year]
	,ec.[Element]
	,pd.[Value]
	,pd.[Unit]
FROM [dbo].[Production_Data] pd
LEFT JOIN [dbo].[Production_AreaCodes] ac on pd.[Area_Code] = ac.[Area_Code]
LEFT JOIN [dbo].[Production_ItemCodes] ic on pd.[Item_Code] = ic.[Item_Code]
LEFT JOIN [dbo].[Production_ElementCodes] ec on pd.Element_Code = ec.[Element_Code]
WHERE pd.[Item_Code] = 515 and pd.[Area_Code] = 33 and ec.[Element] = 'Area harvested'
ORDER BY pd.[Value] ASC;


-- Calculate average annual values for each processing element

SELECT
	 ac.[Area]
	,ic.[Item]
	,ec.[Element]
	,CONVERT(DECIMAL(15,2), AVG(pd.[Value])) AS [Average Value]
	,pd.[Unit]
FROM [dbo].[Production_Data] pd
LEFT JOIN [dbo].[Production_AreaCodes] ac on pd.[Area_Code] = ac.[Area_Code]
LEFT JOIN [dbo].[Production_ItemCodes] ic on pd.[Item_Code] = ic.[Item_Code]
LEFT JOIN [dbo].[Production_ElementCodes] ec on pd.Element_Code = ec.[Element_Code]
WHERE pd.[Item_Code] = 515 and pd.[Area_Code] = 33
GROUP BY ac.[Area], ic.[Item], ec.[Element], pd.[Unit];

GO


/*******************************************************************************************/
/* PART 3                                                                                  */
/*******************************************************************************************/
/* Advanced programming techniques                                                         */
/*******************************************************************************************/

-- Redisplay the apple production data for Canada showing all years

SELECT
	 ac.[Area]
	,ic.[Item]
	,pd.[Year]
	,ec.[Element]
	,pd.[Value]
	,pd.[Unit]
FROM [dbo].[Production_Data] pd
LEFT JOIN [dbo].[Production_AreaCodes] ac ON pd.[Area_Code] = ac.[Area_Code]
LEFT JOIN [dbo].[Production_ItemCodes] ic ON pd.[Item_Code] = ic.[Item_Code]
LEFT JOIN [dbo].[Production_ElementCodes] ec ON pd.Element_Code = ec.[Element_Code]
WHERE pd.[Item_Code] = 515 and pd.[Area_Code] = 33
ORDER BY pd.[Year], ec.[Element];


-- Convert the query into a Common Table Expression (CTE) and display the output

WITH Canada_Apple_CTE ([Area], [Item], [Year], [Element], [Value], [Unit]) AS
(
SELECT
	 ac.[Area] AS [Area]
	,ic.[Item] AS [Item]
	,pd.[Year] AS [Year]
	,ec.[Element] AS [Element]
	,pd.[Value] AS [Value]
	,pd.[Unit] AS [Unit]
FROM [dbo].[Production_Data] pd
LEFT JOIN [dbo].[Production_AreaCodes] ac ON pd.[Area_Code] = ac.[Area_Code]
LEFT JOIN [dbo].[Production_ItemCodes] ic ON pd.[Item_Code] = ic.[Item_Code]
LEFT JOIN [dbo].[Production_ElementCodes] ec ON pd.Element_Code = ec.[Element_Code]
WHERE pd.[Item_Code] = 515 and pd.[Area_Code] = 33
)

SELECT * FROM [Canada_Apple_CTE]
ORDER BY [Year], [Element];


-- Use the same CTE to create a Pivot Table with one row per Year and each Element value in a separate column

WITH Canada_Apple_CTE ([Area], [Item], [Year], [Element], [Value], [Unit]) AS
(
SELECT
	 ac.[Area] AS [Area]
	,ic.[Item] AS [Item]
	,pd.[Year] AS [Year]
	,ec.[Element] AS [Element]
	,pd.[Value] AS [Value]
	,pd.[Unit] AS [Unit]
FROM [dbo].[Production_Data] pd
LEFT JOIN [dbo].[Production_AreaCodes] ac ON pd.[Area_Code] = ac.[Area_Code]
LEFT JOIN [dbo].[Production_ItemCodes] ic ON pd.[Item_Code] = ic.[Item_Code]
LEFT JOIN [dbo].[Production_ElementCodes] ec ON pd.Element_Code = ec.[Element_Code]
WHERE pd.[Item_Code] = 515 and pd.[Area_Code] = 33
)

SELECT * FROM   
	(SELECT [Area], [Item], [Year], [Element], [Value] 
	 FROM [Canada_Apple_CTE]) p  
PIVOT  
	(SUM ([Value])  
	FOR [Element] IN  
	( [Area harvested], [Production], [Yield] )  
	) AS pvt  
	ORDER BY pvt.[Year];

GO

-- Convert the query into an Inline Table Valued Function with parameters
-- to allow different Areas and Items to be selected

CREATE FUNCTION [dbo].[Select_Area_and_Item](@areaname AS NVARCHAR(MAX), @itemname AS NVARCHAR(MAX))
RETURNS TABLE
AS
RETURN
(
SELECT
	 ac.[Area] AS [Area]
	,ic.[Item] AS [Item]
	,pd.[Year] AS [Year]
	,ec.[Element] AS [Element]
	,pd.[Value] AS [Value]
	,pd.[Unit] AS [Unit]
FROM [dbo].[Production_Data] pd
LEFT JOIN [dbo].[Production_AreaCodes] ac ON pd.[Area_Code] = ac.[Area_Code]
LEFT JOIN [dbo].[Production_ItemCodes] ic ON pd.[Item_Code] = ic.[Item_Code]
LEFT JOIN [dbo].[Production_ElementCodes] ec ON pd.Element_Code = ec.[Element_Code]
WHERE ic.[Item] = @itemname and ac.[Area] = @areaname
);
GO


-- Use function to select different Areas and Items to display

SELECT * FROM [dbo].[Select_Area_and_Item]('Mexico', 'Strawberries')
ORDER BY [Year], [Element];

SELECT * FROM [dbo].[Select_Area_and_Item]('Caribbean', 'Sugar cane')
ORDER BY [Year], [Element];

SELECT * FROM [dbo].[Select_Area_and_Item]('World', 'Cattle')
ORDER BY [Year], [Element];








/*******************************************************************************************/
/* PART 4                                                                                  */
/*******************************************************************************************/
/* Clean up temporary objects                                                              */
/*******************************************************************************************/

DROP TABLE IF EXISTS [tempdb].[dbo].[Reccount];
DROP TABLE IF EXISTS [tempdb].[dbo].[Tablelist]; 
DROP FUNCTION IF EXISTS [dbo].[Select_Area_and_Item];
