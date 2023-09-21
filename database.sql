use master
go

if exists(select * from sys.databases where name = N'Wisdom')
begin
	drop database[FaceRecognition] 
end

create database [FaceRecognition];
go
use FaceRecognition

create table [User] (
   Id					int                  identity,
   FullName             nvarchar(40)         not null,
   City                 nvarchar(40)         null,
   Country              nvarchar(40)         null,
   Phone                nvarchar(20)         null,
   constraint PK_User primary key (Id)
)
go

create table Attendance(
   Id					int                  identity,
   FullName				nvarchar(40)         not null,
   DateIn				datetime			 not null,
   TimeIn				time				 not null,
   UserId				int					 null
   constraint PK_Attendance primary key (Id)
)
go

insert into FaceRecognition.dbo.[User] (FullName, City, Country, Phone) values('Trump', 'New York', 'USA', '0123456789')
insert into FaceRecognition.dbo.[User] (FullName, City, Country, Phone) values('Obama', 'Honolulu', 'USA', '9876543201')
insert into FaceRecognition.dbo.[User] (FullName, City, Country, Phone) values('Putin', ' St. Petersburg', 'Russia', '0123123123')





